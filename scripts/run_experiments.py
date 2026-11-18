
#!/usr/bin/env python3
"""
kANNolo Experiment Runner

Supports:
- hnsw_build / hnsw_search            HNSW indexes (dense and sparse)
- hnsw_rerank_search                   Two-stage reranking (multivec)
- ivf_build / ivf_search               IVF indexes (dense; Euclidean or dot product)

IVFFlat TOML parameters
  [indexing_parameters]
    n-clusters      number of k-means clusters
    hnsw            true / false  (centroid index type)
    m-hnsw          HNSW M        (ignored when hnsw = false)
    ef-construction HNSW efc      (ignored when hnsw = false)
    residuals       true / false
    m-pq            PQ subspaces  (omit for plain f32 clusters)
    kmeans-solver   (optional, lloyd / emeans; default: vectorium's default)
    kmeans-epochs   (optional, default 25 for lloyd, 5 for emeans)
    kmeans-n-redo   (optional, default 1)

  [querying_parameters]
    n-probe         number of clusters to probe
    ef-search       HNSW ef_search (ignored when hnsw = false)
    lambda          early-term lambda (ignored when hnsw = false)
"""

import re 
import os
import sys
import time
import socket
import argparse
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import ir_measures
import toml
import psutil

from termcolor import colored

def parse_toml(filename):
    """Parse the TOML configuration file."""
    try:
        return toml.load(filename)
    except Exception as e:
        print(f"Error reading the TOML file: {e}")
        return None


def get_git_info(experiment_dir):
    """Get Git repository information and save it to git.output."""
    print()
    print(colored("Git info", "green"))
    git_output_file = os.path.join(experiment_dir, "git.output")

    try:
        with open(git_output_file, "w") as git_output:
            # Get current branch
            branch_process = subprocess.Popen("git rev-parse --abbrev-ref HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            branch_name = branch_process.stdout.read().decode().strip()
            branch_process.wait()

            # Get current commit id
            commit_process = subprocess.Popen("git rev-parse HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commit_id = commit_process.stdout.read().decode().strip()
            commit_process.wait()

            # Write to git.output
            git_output.write(f"Current Branch: {branch_name}\n")
            git_output.write(f"Commit ID: {commit_id}\n")
            print(f"Current Branch: {branch_name}")
            print(f"Commit ID: {commit_id}")

    except Exception as e:
        print("An error occurred while retrieving Git information:", e)
        sys.exit(1)


def compile_rust_code(configs, experiment_dir):
    """Compile the Rust code and save output."""
    print()
    print(colored("Compiling the Rust code", "green"))
    
    # The binaries are behind the `cli` feature; hnsw_rerank_search also needs `multivec`.
    features = "cli"
    if "rerank" in configs.get("query-command", ""):
        features = "cli,multivec"
    default_compile_command = (
        f"RUSTFLAGS='-C target-cpu=native' cargo build --release --features {features}"
    )
    compile_command = configs.get("compile-command", default_compile_command)

    compilation_output_file = os.path.join(experiment_dir, "compiler.output")

    try:
        print("Compiling Rust code with", compile_command)
        with open(compilation_output_file, "w") as comp_output:
            compile_process = subprocess.Popen(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(compile_process.stdout.readline, b''):
                decoded_line = line.decode()
                print(decoded_line, end='')  # Print each line as it is produced
                comp_output.write(decoded_line)  # Write each line to the output file
            compile_process.stdout.close()
            compile_process.wait()

        if compile_process.returncode != 0:
            print("Rust compilation failed.")
            sys.exit(1)
        print("Rust code compiled successfully.")

    except Exception as e:
        print()
        print(colored("ERROR: Problems during Rust compilation:", "red"), e)
        sys.exit(1)


DEFAULT_GRAPH_TYPE = "standard"


def get_graph_type(configs, query_config=None):
    """Resolve the graph type of a query subsection, falling back to the top-level value."""
    default = configs.get("graph-type", DEFAULT_GRAPH_TYPE)
    if query_config is None:
        return default
    return query_config.get("graph-type", default)


def collect_graph_types(configs):
    """Every graph type the experiment needs an index for, in first-seen order."""
    if "query" not in configs:
        return [get_graph_type(configs)]

    graph_types = []
    for query_config in configs["query"].values():
        graph_type = get_graph_type(configs, query_config)
        if graph_type not in graph_types:
            graph_types.append(graph_type)
    return graph_types or [get_graph_type(configs)]


def get_index_filename(base_filename, configs, graph_type=None):
    """Generate the index filename based on the provided parameters."""
    name = []

    name.append(base_filename)

    # Check if pq_parameters and pq-subspaces exist
    if "pq_parameters" in configs and "pq-subspaces" in configs["pq_parameters"]:
        name.append(f"pq-subspaces_{configs['pq_parameters']['pq-subspaces']}")

    # Append indexing parameters
    name += sorted(f"{k}_{v}" for k, v in configs["indexing_parameters"].items())

    # Only non-default graph types go in the name, so "standard" indexes keep their filename.
    if graph_type is not None and graph_type != DEFAULT_GRAPH_TYPE:
        name.append(f"graph-type_{graph_type}")

    return "_".join(str(l) for l in name)


def _bool_flag(value):
    """Convert a Python bool (possibly from TOML) to a lowercase string for CLI flags."""
    return str(value).lower()


def _build_ivf_params(configs, input_file, output_file):
    """Assemble ivf_build command-line parameters."""
    ip = configs["indexing_parameters"]
    params = [
        configs["build-command"],
        f"--data-file {input_file}",
        f"--output-file {output_file}",
        f"--n-clusters {ip['n-clusters']}",
    ]
    # Boolean flags: pass bare flag when true, omit entirely when false (clap default).
    if "value-type" in configs:
        params.append(f"--value-type {configs['value-type']}")
    # metric in [indexing_parameters] is the primary source; top-level distance is a fallback
    dist = ip.get("metric") or configs.get("distance")
    if dist:
        params.append(f"--distance {dist}")
    if ip.get("hnsw", False):
        params.append("--hnsw")
        params.append(f"--m-hnsw {ip.get('m-hnsw', 32)}")
        params.append(f"--ef-construction {ip.get('ef-construction', 200)}")
    if ip.get("residuals", False):
        params.append("--residuals")
    if "m-pq" in ip:
        params.append(f"--m-pq {ip['m-pq']}")
    if "kmeans-solver" in ip:
        params.append(f"--kmeans-solver {ip['kmeans-solver']}")
    if "kmeans-epochs" in ip:
        params.append(f"--kmeans-epochs {ip['kmeans-epochs']}")
    elif "kmeans-n-iter" in ip:  # older configs; same value (one Lloyd iteration = one epoch)
        params.append(f"--kmeans-epochs {ip['kmeans-n-iter']}")
    if "kmeans-n-redo" in ip:
        params.append(f"--kmeans-n-redo {ip['kmeans-n-redo']}")
    if "kmeans-sample-size" in ip:
        params.append(f"--kmeans-sample-size {ip['kmeans-sample-size']}")
    if ip.get("kmeans-hnsw", False):
        params.append("--kmeans-hnsw")
    if ip.get("kmeans-spherical", False):
        params.append("--kmeans-spherical")
    return params


# hnsw_build prints two different build-time lines, so match on their shared shape.
BUILD_TIME_PATTERN = re.compile(r"^Time to build\b.*?:\s*(\d+)\s*s \(before serializing\)\s*$")


def build_index(configs, experiment_dir, graph_type=None):
    """Build the index using the provided configuration."""
    if configs.get("dataset-type") == "multivector":
        raise ValueError("multivector support was removed; update the experiment config to use dense or sparse vectors.")
    input_file =  os.path.join(configs["folder"]["data"], configs["filename"]["dataset"])
    index_folder = configs["folder"]["index"]

    if graph_type is None:
        graph_type = get_graph_type(configs)

    os.makedirs(index_folder, exist_ok=True)
    output_file = os.path.join(index_folder, get_index_filename(configs["filename"]["index"], configs, graph_type))

    print()
    print(colored(f"Dataset filename:", "blue"), input_file)
    print(colored(f"Index filename:", "blue"), output_file)

    build_command = configs.get("build-command", None)
    if not build_command:
        raise ValueError("Build command must be specified!!!")

    is_ivf = "ivf_build" in build_command

    if is_ivf:
        command_and_params = _build_ivf_params(configs, input_file, output_file)
    else:
        metric = configs["indexing_parameters"]["metric"]
        if metric == "l2":
            print(colored("Warning: metric 'l2' is deprecated; use 'euclidean'.", "yellow"))
            metric = "euclidean"
        elif metric == "ip":
            print(colored("Warning: metric 'ip' is deprecated; use 'dotproduct'.", "yellow"))
            metric = "dotproduct"

        ef_construction = configs["indexing_parameters"].get(
            "ef-construction",
            configs["indexing_parameters"].get("efc"),
        )
        if ef_construction is None:
            raise ValueError("indexing_parameters must include 'ef-construction' (or legacy 'efc').")
        if "efc" in configs["indexing_parameters"] and "ef-construction" not in configs["indexing_parameters"]:
            print(colored("Warning: 'efc' is deprecated; use 'ef-construction'.", "yellow"))

        command_and_params = [
            build_command,
            f"--data-file {input_file}",
            f"--output-file {output_file}",
            f"--m {configs['indexing_parameters']['m']}",
            f"--ef-construction {ef_construction}",
            f"--distance {metric}",
        ]

        # Add new unified binary parameters
        if "dataset-type" in configs:
            command_and_params.append(f"--dataset-type {configs['dataset-type']}")
        if "value-type" in configs:
            command_and_params.append(f"--value-type {configs['value-type']}")
        if configs.get("dataset-type") == "sparse" and "component-type" in configs:
            command_and_params.append(f"--component-type {configs['component-type']}")
        if "encoder" in configs:
            command_and_params.append(f"--encoder {configs['encoder']}")
        command_and_params.append(f"--graph-type {graph_type}")
        # If there is a section [pq_parameters] in the configuration file, add the parameters to the command
        if "pq_parameters" in configs:
            for k, v in configs["pq_parameters"].items():
                command_and_params.append(f"--{k} {v}")

    command = ' '.join(command_and_params)

    # Print the command that will be executed
    print()
    print(colored(f"Indexing (graph-type: {graph_type})", "green"))
    print(colored(f"Indexing command:", "blue"), command)

    building_output_file = os.path.join(experiment_dir, f"building_{graph_type}.output")

    # Build the index and display output in real-time
    print(colored("Building index...", "yellow"))
    building_time = 0

    with open(building_output_file, "w") as build_output:
        build_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(build_process.stdout.readline, b''):
            decoded_line = line.decode()
            print(decoded_line, end='')  # Print each line as it is produced
            build_output.write(decoded_line)  # Write each line to the output file
            match = BUILD_TIME_PATTERN.match(decoded_line.strip())
            if match:
                building_time = int(match.group(1))

        build_process.stdout.close()
        build_process.wait()

    if build_process.returncode != 0:
        print(colored("ERROR: Indexing failed!", "red"))
        sys.exit(1)

    print(colored(f"Index built successfully in {building_time} secs!", "yellow"))
    return building_time


def compute_metric(configs, output_file, gt_file, metric): 

    if metric == None or metric == "":
        print("No metric specified. Skipping evaluation.")
        return None

    column_names = ["query_id", "doc_id", "rank", "score"]
    gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
    res_pd = pd.read_csv(output_file, sep='\t', names=column_names)
    
    query_ids_path = os.path.join(configs['folder']['data'], configs['filename']['query_ids'])
    queries_ids = np.load(query_ids_path, allow_pickle=True)

    document_ids_path = os.path.join(configs['folder']['data'], configs['filename']['doc_ids'])
    doc_ids = np.load(os.path.realpath(document_ids_path), allow_pickle=True)
    
    gt_pd['query_id'] = gt_pd['query_id'].apply(lambda x: queries_ids[x])
    res_pd['query_id'] = res_pd['query_id'].apply(lambda x: queries_ids[x])
    
    gt_pd['doc_id'] = gt_pd['doc_id'].apply(lambda x: doc_ids[x])
    res_pd['doc_id'] = res_pd['doc_id'].apply(lambda x: doc_ids[x])
    
    qrels_path = configs['folder']['qrels_path']
    
    df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "useless", "doc_id", "relevance"])
    #if "nq" in configs['name']: # the order of the fields in nq is different. 
    if len(pd.unique(df_qrels['useless'])) != 1:
        df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "doc_id", "relevance", "useless"])

    gt_pd['doc_id'] = gt_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    res_pd['doc_id'] = res_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    
    gt_pd['query_id'] = gt_pd['query_id'].astype(df_qrels.query_id.dtype)
    res_pd['query_id'] = res_pd['query_id'].astype(df_qrels.query_id.dtype)
    
    # pytrec_eval (used by ir_measures) expects string keys for query ids and doc ids.
    # Ensure all ids are strings to avoid TypeError: Expected string as key.
    df_qrels['query_id'] = df_qrels['query_id'].astype(str)
    df_qrels['doc_id'] = df_qrels['doc_id'].astype(str)
    gt_pd['query_id'] = gt_pd['query_id'].astype(str)
    res_pd['query_id'] = res_pd['query_id'].astype(str)
    gt_pd['doc_id'] = gt_pd['doc_id'].astype(str)
    res_pd['doc_id'] = res_pd['doc_id'].astype(str)
    
    ir_metric = ir_measures.parse_measure(metric)
    
    metric_val = ir_measures.calc_aggregate([ir_metric], df_qrels, res_pd)[ir_metric]
    metric_gt = ir_measures.calc_aggregate([ir_metric], df_qrels, gt_pd)[ir_metric]
    
    print(f"Metric of the run: {ir_metric}: {metric_val}")
    print(f"Metric of the gt : {ir_metric}: {metric_gt}")
    
    return metric_val
    

def compute_accuracy(query_file, gt_file):
    # if files are csv
    if gt_file.endswith(".csv") or gt_file.endswith(".tsv"):
        column_names = ["query_id", "doc_id", "rank", "score"]
        if gt_file.endswith(".csv"):
            gt_pd = pd.read_csv(gt_file, sep=',', names=column_names)
            res_pd = pd.read_csv(query_file, sep=',', names=column_names)
        else:
            gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
            res_pd = pd.read_csv(query_file, sep='\t', names=column_names)

        # Group both dataframes by 'query_id' and get unique 'doc_id' sets
        gt_pd_groups = gt_pd.groupby('query_id')['doc_id'].apply(set)
        res_pd_groups = res_pd.groupby('query_id')['doc_id'].apply(set)

        # Compute the intersection size for each query_id in both dataframes
        intersections_size = {
            query_id: len(gt_pd_groups[query_id] & res_pd_groups[query_id]) if query_id in res_pd_groups else 0
            for query_id in gt_pd_groups.index
        }

        # Computes total number of results in the groundtruth
        total_results = len(gt_pd)
        total_intersections = sum(intersections_size.values())

    elif gt_file.endswith(".npy"):
        # Read csv results and transform to numpy array
        column_names = ["query_id", "doc_id", "rank", "score"]
        res_pd = pd.read_csv(query_file, sep='\t', names=column_names)

        # Read npy groundtruth
        doc_ids = np.load(gt_file, allow_pickle=True)

        # Use groupby to handle variable result counts per query (e.g. when
        # adaptive n_probe or a small n_probe returns fewer than k results for
        # some queries, making the total non-divisible by k and breaking reshape).
        k = res_pd.groupby('query_id').size().max()
        grouped = res_pd.groupby('query_id')['doc_id']
        total_results = 0
        total_intersections = 0
        for query_id, res_ids in grouped:
            res_arr = res_ids.to_numpy()
            gt_arr = doc_ids[query_id][:k]
            total_intersections += len(np.intersect1d(res_arr, gt_arr))
            total_results += k   # denominator is always k (ideal), not actual returned
    else:
        raise ValueError("Groundtruth file must be in csv or numpy format!!!")
        
    return round((total_intersections/total_results) * 100, 3)


def query_execution(configs, query_config, experiment_dir, subsection_name, subsection_index=None, total_subsections=None):
    """Execute a query based on the provided configuration."""
    if configs.get("dataset-type") == "multivector":
        raise ValueError("multivector support was removed; update the experiment config to use dense or sparse vectors.")

    query_command = configs.get("query-command", None)
    if not query_command:
        raise ValueError("Query command must be specified!!!")

    # The graph type selects which index this subsection searches.
    graph_type = get_graph_type(configs, query_config)
    index_file = os.path.join(
        configs["folder"]["index"],
        get_index_filename(configs["filename"]["index"], configs, graph_type),
    )
    print(f"Searching index at: {index_file} (graph-type: {graph_type})")

    query_file = os.path.join(configs["folder"]["data"], configs["filename"]["queries"])
    output_file = os.path.join(experiment_dir, f"results_{subsection_name}")

    is_ivf = "ivf_search" in query_command
    is_reranking_binary = "rerank" in query_command

    numa_prefix = configs['settings']['NUMA'] if "NUMA" in configs['settings'] else ""

    if is_ivf:
        ip = configs["indexing_parameters"]
        command_and_params = [
            numa_prefix,
            query_command,
            f"--index-file {index_file}",
            f"--query-file {query_file}",
            f"--k {configs['settings']['k']}",
            f"--n-probe {query_config['n-probe']}",
            f"--output-path {output_file}",
        ]
        if "value-type" in configs:
            command_and_params.append(f"--value-type {configs['value-type']}")
        dist = ip.get("metric") or configs.get("distance")
        if dist:
            command_and_params.append(f"--distance {dist}")
        # Boolean flags: pass bare flag when true, omit when false (clap default).
        if ip.get("hnsw", False):
            command_and_params.append("--hnsw")
        # No --residuals here: ivf_search reads residual-vs-plain back from the index.
        if "m-pq" in ip:
            command_and_params.append(f"--m-pq {ip['m-pq']}")
        # ef-search and lambda are only meaningful when using HNSW centroids.
        if ip.get("hnsw", False):
            if "ef-search" in query_config:
                command_and_params.append(f"--ef-search {query_config['ef-search']}")
            if "lambda" in query_config:
                command_and_params.append(f"--lambda {query_config['lambda']}")
        if "num-runs" in configs["settings"]:
            command_and_params.append(f"--num-runs {configs['settings']['num-runs']}")
    else:
        metric = configs["indexing_parameters"]["metric"]
        if metric == "l2":
            print(colored("Warning: metric 'l2' is deprecated; use 'euclidean'.", "yellow"))
            metric = "euclidean"
        elif metric == "ip":
            print(colored("Warning: metric 'ip' is deprecated; use 'dotproduct'.", "yellow"))
            metric = "dotproduct"

        command_and_params = [
            numa_prefix,
            query_command,
            f"--index-file {index_file}",
            f"--query-file {query_file}",
            f"--k {configs['settings']['k']}",
            f"--ef-search {query_config['ef-search']}",
            f"--distance {metric}",
            f"--output-path {output_file}",
        ]

    if not is_ivf and not is_reranking_binary:
        if "dataset-type" in configs:
            command_and_params.append(f"--dataset-type {configs['dataset-type']}")
        if "value-type" in configs:
            command_and_params.append(f"--value-type {configs['value-type']}")
        if configs.get("dataset-type") == "sparse" and "component-type" in configs:
            command_and_params.append(f"--component-type {configs['component-type']}")
        if "encoder" in configs:
            command_and_params.append(f"--encoder {configs['encoder']}")
        command_and_params.append(f"--graph-type {graph_type}")
        if "num-runs" in configs["settings"]:
            command_and_params.append(f"--num-runs {configs['settings']['num-runs']}")

        # Add early termination parameters if specified
        early_termination = query_config.get("early-termination", "none")
        command_and_params.append(f"--early-termination {early_termination}")
        if "lambda" in query_config:
            command_and_params.append(f"--lambda {query_config['lambda']}")

        # Add PQ-specific parameters if needed
        if "pq_parameters" in configs and "pq-subspaces" in configs['pq_parameters']:
            command_and_params.append(f"--pq-subspaces {configs['pq_parameters']['pq-subspaces']}")

    # Add multivector reranking parameters if present (only for hnsw_rerank_search)
    if is_reranking_binary:
        multivec_data_dir = configs["folder"].get("multivec_data", "")
        if not multivec_data_dir:
            raise ValueError("multivec_data folder must be specified in [folder] section for reranking")
        
        # Pass the data folder directly (binary will look for documents.npy, doclens.npy, queries.npy inside)
        command_and_params.append(f"--multivec-data-folder {multivec_data_dir}")
        
        # Get quantizer type and PQ subspaces from [multivector] section if present
        quantizer_type = "plain"  # Default to plain
        pq_subspaces = None
        
        if "multivector" in configs:
            quantizer_type = configs["multivector"].get("quantizer", "plain")
            pq_subspaces = configs["multivector"].get("pq-subspaces", None)
        
        command_and_params.append(f"--multivector-quantizer {quantizer_type}")
        
        if quantizer_type == "two-levels":
            if pq_subspaces is None:
                raise ValueError("pq-subspaces must be specified in [multivector] section for 'two-levels' quantizer")
            command_and_params.append(f"--pq-subspaces {pq_subspaces}")
        
        # Extract reranking parameters from query_config subsection
        k_candidates = query_config.get("k_candidates", 100)
        command_and_params.append(f"--k-candidates {k_candidates}")
        
        if "alpha" in query_config and query_config["alpha"] is not None:
            command_and_params.append(f"--alpha {query_config['alpha']}")
        
        if "beta" in query_config and query_config["beta"] is not None:
            command_and_params.append(f"--beta {query_config['beta']}")
        
        # Add early termination parameters for first-stage search
        early_termination = query_config.get("early-termination", "none")
        command_and_params.append(f"--early-termination {early_termination}")
        if "lambda" in query_config:
            command_and_params.append(f"--lambda {query_config['lambda']}")

    command = " ".join(command_and_params)

    print(f"Executing query for subsection '{subsection_name}' with command:")
    print(command)

    # Define log and output files
    log_output_file = os.path.join(experiment_dir, f"log_{subsection_name}")
    output_file = os.path.join(experiment_dir, f"results_{subsection_name}")

    pattern = r"Total: (\d+) bytes"  # Pattern to match the total memory usage
    bits_per_edge_pattern = r"\[######\] ([\d.]+) bits/edge"

    query_time = 0
    # Keep the last value seen: resetting inside the loop would clear it on the next line.
    memory_usage = 0
    bits_per_edge = 0.0
    # Run the query and display output in real-time
    if subsection_index is not None and total_subsections is not None:
        print(f"Running query for subsection: {subsection_name} out of {total_subsections}...")
    else:
        print(f"Running query for subsection: {subsection_name}...")
    with open(log_output_file, "w") as log:
        query_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(query_process.stdout.readline, b''):
            decoded_line = line.decode()
            if decoded_line.startswith("[######] Average Query Time"):
                match = re.search(r"Average Query Time: (\d+)", decoded_line)
                if match:
                    query_time = int(match.group(1))

            match = re.search(pattern, decoded_line)
            if match:
                memory_usage = int(match.group(1))

            match = re.search(bits_per_edge_pattern, decoded_line)
            if match:
                bits_per_edge = float(match.group(1))

            print(decoded_line, end='')  # Print each line as it is produced
            log.write(decoded_line)  # Write each line to the output file
        query_process.stdout.close()
        query_process.wait()

    if query_process.returncode != 0:
        print(f"Query execution for subsection '{subsection_name}' failed.")
        sys.exit(1)

    print(f"Query for subsection '{subsection_name}' executed successfully.")

    gt_file = os.path.join(configs['folder']['data'], configs['filename']['groundtruth'])
    metric = configs['settings']['metric']
    accuracy = compute_accuracy(output_file, gt_file)
    metric_value = compute_metric(configs, output_file, gt_file, metric)

    # Accuracy is computed after the search process exits, so append it to the log by hand.
    print(f"Accuracy: {accuracy}")
    with open(log_output_file, "a") as log:
        log.write(f"Accuracy: {accuracy}\n")
        if metric_value is not None:
            log.write(f"Metric ({metric}): {metric_value}\n")

    return query_time, accuracy, metric_value, memory_usage, bits_per_edge, graph_type


def get_machine_info(configs, experiment_folder):
    machine_info_file = os.path.join(experiment_folder, "machine.output")
    machine_info = open(machine_info_file, "w")

    date = datetime.now()
    machine = socket.gethostname()
    cpu = psutil.cpu_percent(interval=1)
    
    memory_free = psutil.virtual_memory().free // (1024 ** 3)
    memory_avail = psutil.virtual_memory().available // (1024 ** 3)
    memory_total = psutil.virtual_memory().total // (1024 ** 3)
    
    load = str(psutil.getloadavg())[1:-1]
    num_cpus = psutil.cpu_count()
    
    machine_info.write(f"----------------------\n")
    machine_info.write(f"Hardware configuration\n")
    machine_info.write(f"----------------------\n")
    machine_info.write(f"Date: {date}\n")
    machine_info.write(f"Machine: {machine}\n")
    machine_info.write(f"CPU usage (%): {cpu}\n")
    machine_info.write(f"Machine load: {load}\n")
    machine_info.write(f"Memory (free, GiB): {memory_free}\n")
    machine_info.write(f"Memory (avail, GiB): {memory_avail}\n")
    machine_info.write(f"Memory (total, GiB): {memory_total}\n")
    
    print()
    print(colored("Hardware configuration", "green"))
    print(f"Date: {date}")
    print(f"Machine: {machine}")
    print(f"CPU usage (%): {cpu}")
    print(f"Machine load: {load}")
    print(f"Memory (free, GiB): {memory_free}")
    print(f"Memory (avail, GiB): {memory_avail}")
    print(f"Memory (total, GiB): {memory_total}")
    print(f"for detailed information, check the hardware log file: {machine_info_file}")

    machine_info.write(f"\n---------------------\n")
    machine_info.write(f"cpufreq configuration\n")
    machine_info.write(f"---------------------\n")

    command_governor = 'cpufreq-info | grep "performance" | grep -v "available" | wc -l'
    governor = subprocess.Popen(command_governor, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    governor.wait()

    for line in iter(governor.stdout.readline, b''):
        cpus_with_performance_governor = int(line.decode())
        machine_info.write(f'Number of CPUs with governor set to "performance" (should be equal to the number of CPUs below): {cpus_with_performance_governor}\n')

    # checking if the hardware looks well configured...
    if (num_cpus != cpus_with_performance_governor):
        print()
        print(colored("ERROR: Problems with hardware configuration found!", "red"))
        print(colored("Your CPU is not set to performance mode. Please, run `cpufreq-info` for more details.", "red"))
        print()

    machine_info.write(f"\n-----------------\n")
    machine_info.write(f"CPU configuration\n")
    machine_info.write(f"-----------------\n")

    command_cpu = 'lscpu'
    cpu = subprocess.Popen(command_cpu, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cpu.wait()

    for line in iter(cpu.stdout.readline, b''):
        decoded_line = line.decode()
        machine_info.write(decoded_line)

    if ("NUMA" in configs['settings']):
        machine_info.write(f"\n------------------------------------------------------------------------------\n")
        machine_info.write(f"NUMA execution command (check if CPU IDs corresponds to physical ones (no HT))\n")
        machine_info.write(f"------------------------------------------------------------------------------\n")
        machine_info.write(f'Shell command: "{configs["settings"]["NUMA"]}"\n')

        machine_info.write(f"\n------------------\n")
        machine_info.write(f"NUMA configuration\n")
        machine_info.write(f"------------------\n")

        command_numa = 'numactl --hardware'
        numa = subprocess.Popen(command_numa, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        numa.wait()

        for line in iter(numa.stdout.readline, b''):
            decoded_line = line.decode()
            machine_info.write(decoded_line)

    machine_info.close()
    return


def run_experiment(config_data):
    """Run the kannolo experiment based on the provided configuration."""

     # Get the experiment name from the configuration
    experiment_name = config_data.get("name")
    print(f"Running experiment:", colored(experiment_name, "green"))

    for k, v in config_data["folder"].items():
        if v.startswith("~"):
            v = os.path.expanduser(v)
            config_data["folder"][k] = v

   #print(config_data)

    # Create an experiment folder with date and hour
    timestamp  = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    experiment_folder = os.path.join(config_data["folder"]["experiment"], f"{experiment_name}_{timestamp}")

    os.makedirs(experiment_folder, exist_ok=True)

    # Dump the configuration settings to a TOML file
    with open(os.path.join(experiment_folder, "experiment_config.toml"), 'w') as report_file:
        report_file.write(toml.dumps(config_data))

    # Retrieving hardware information
    get_machine_info(config_data, experiment_folder)

    # Store the output of the Rust compilation and index building processes
    get_git_info(experiment_folder)
    
    compile_rust_code(config_data, experiment_folder)

    # Graph type is a build-time property: an experiment mixing them needs one index each.
    graph_types = collect_graph_types(config_data)
    building_times = {}
    if config_data['settings']['build']:
        if len(graph_types) > 1:
            print(f"Building {len(graph_types)} indexes, one per graph type: {', '.join(graph_types)}")
        for graph_type in graph_types:
            building_times[graph_type] = build_index(config_data, experiment_folder, graph_type)
    else:
        print("Index is already built!")

    metric = config_data['settings']['metric']
    print(f"Evaluation runs with metric {metric}")

    # Execute queries for each subsection under [query]
    with open(os.path.join(experiment_folder, "report.tsv"), 'w') as report_file:
        # Column order matches the reference CIKM 2026 reports.
        metric_header = f"\t{metric}" if metric != "" else ""
        report_file.write(f"Subsection\tQuery Time (microsecs)\tBits/Edge\tAccuracy{metric_header}\tMemory Usage (Bytes)\tBuilding Time (secs)\n")
        if 'query' in config_data:
            total_subsections = len(config_data['query'])
            for subsection_index, (subsection, query_config) in enumerate(config_data['query'].items(), start=1):
                query_time, recall, metric_value, memory_usage, bits_per_edge, graph_type = query_execution(config_data, query_config, experiment_folder, subsection, subsection_index, total_subsections)
                # Report the build time of the index this subsection actually searched.
                building_time = building_times.get(graph_type, 0)
                if metric_value is not None:
                    report_file.write(f"{subsection}\t{query_time}\t{bits_per_edge}\t{recall}\t{metric_value}\t{memory_usage}\t{building_time}\n")
                else:
                    report_file.write(f"{subsection}\t{query_time}\t{bits_per_edge}\t{recall}\t{memory_usage}\t{building_time}\n")

def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print()
        print(colored("ERROR: Configuration data is empty.", "red"))
        sys.exit(1)
    run_experiment(config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a kANNolo experiment on a dataset and query it.")
    parser.add_argument("--exp", required=True, help="Path to the experiment configuration TOML file.")
    args = parser.parse_args()

    main(args.exp)
    sys.exit(0)
