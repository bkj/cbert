// cbert.cu

#ifndef MAIN_AUCTION
#define MAIN_AUCTION

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>

#include "cbert.cuh"
#include "bidding.cuh"
#include "assignment.cuh"
#include "timer.cuh"

// --
// Define constants

#ifndef __RUN_VARS
#define __RUN_VARS
#define AUCTION_MAX_EPS 1.0 // Larger values mean solution is more approximate
#define AUCTION_MIN_EPS 1.0
#define AUCTION_FACTOR  0.0
#define NUM_RUNS        1
#endif

// struct Min2Op {
//     __device__ __forceinline__
//     Entry operator()(const Entry &a, const Entry &b) const {
//         float best_val, next_best_val;
//         if(a.best_val < b.best_val) {
//             best_val = a.best_val;
//             next_best_val = min(a.next_best_val, b.best_val);
//         } else {
//             best_val = b.best_val;
//             next_best_val = min(b.next_best_val, a.best_val);
//         }
//         return (Entry){0, best_val, next_best_val, 0};
//     }
// };

extern "C" {

int run_auction(
    int    num_nodes,
    int    num_edges,

    float* h_data,      // data
    int*   h_offsets,   // offsets for items
    int*   h_columns,

    int*   h_person2item, // results

    float auction_max_eps,
    float auction_min_eps,
    float auction_factor,

    int num_runs,
    int verbose,

    int cub_bidding
)
{
    printf("cub_bidding=%d\n", cub_bidding);

    int node_blocks = 1 + num_nodes / THREADS;
    int  h_num_assigned;

    AuctionData ad;
    cudaMalloc((void **)&ad.data,          num_edges             * sizeof(float));
    cudaMalloc((void **)&ad.columns,       num_edges             * sizeof(float));
    cudaMalloc((void **)&ad.offsets,       (num_nodes + 1)       * sizeof(int));
    cudaMalloc((void **)&ad.person2item,   num_nodes             * sizeof(int));
    cudaMalloc((void **)&ad.item2person,   num_nodes             * sizeof(int));
    cudaMalloc((void **)&ad.bids,          num_nodes * num_nodes * sizeof(float));
    cudaMalloc((void **)&ad.prices,        num_nodes             * sizeof(float));
    cudaMalloc((void **)&ad.sbids,         num_nodes             * sizeof(int));
    cudaMalloc((void **)&ad.num_assigned,           1 * sizeof(int)) ;
    cudaMalloc((void **)&ad.rand,          num_nodes * num_nodes * sizeof(float)) ;

    cudaMalloc((void **)&ad.flags,                    num_nodes * sizeof(int));
    cudaMalloc((void **)&ad.num_unassigned,                   1 * sizeof(int));
    cudaMalloc((void **)&ad.unassigned_offsets_start, num_nodes * sizeof(int));
    cudaMalloc((void **)&ad.unassigned_offsets_end,   num_nodes * sizeof(int));

    cudaMalloc((void**)&ad.entry_array, num_edges * sizeof(Entry));

    cudaMemset(ad.num_assigned, 0, 1 * sizeof(int));

    // --
    // Copy from host to device

    cudaMemcpy(ad.data,    h_data,    num_edges       * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ad.columns, h_columns, num_edges       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(ad.offsets, h_offsets, (num_nodes + 1) * sizeof(int),   cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.Start();

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123);
    curandGenerateUniform(gen, ad.rand, num_nodes * num_nodes);

    __make_entry_array<<<node_blocks, THREADS>>>(ad.entry_array, ad.offsets, ad.columns, ad.data, ad.rand, num_nodes);

    for(int run_num = 0; run_num < num_runs; run_num++) {

        cudaMemset(ad.prices, 0.0, num_nodes * sizeof(float));

        h_num_assigned = 0;
        cudaMemset(ad.person2item,   -1, num_nodes * sizeof(int));
        cudaMemset(ad.item2person,   -1, num_nodes * sizeof(int));
        cudaMemset(ad.num_assigned,   0, 1         * sizeof(int));

        int iter = 0;
        while(h_num_assigned < num_nodes){
            cudaMemset(ad.bids,  0, num_nodes * num_nodes * sizeof(float));
            cudaMemset(ad.sbids, 0, num_nodes * sizeof(int));

            run_bidding(num_nodes, ad, auction_max_eps, cub_bidding);
            run_assignment(num_nodes, ad, false);

            cudaMemcpy(&h_num_assigned, ad.num_assigned, sizeof(int) * 1, cudaMemcpyDeviceToHost);

            // printf("h: iter=%d | h_num_assigned=%d\n", iter, h_num_assigned);

            iter++;
        }
     }
    timer.Stop();

    cudaMemcpy(h_person2item, ad.person2item, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

    return timer.ElapsedMillis();
}

}

int main(int argc, char **argv)
{

    int use_cub = atoi(argv[1]);

    std::cerr << "loading ./graph" << std::endl;
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<float> data;

    std::ifstream input_file("graph", std::ios_base::in);
    int src, dst;
    float val;

    int last_src = -1;
    int i = 0;
    while(input_file >> src >> dst >> val) {
        if(src != last_src) {
            offsets.push_back(i);
            last_src = src;
        }
        columns.push_back(dst);
        data.push_back(val);
        i++;
    }
    offsets.push_back(i);

    int* h_offsets = &offsets[0];
    int* h_columns = &columns[0];
    float* h_data  = &data[0];

    int num_nodes = offsets.size() - 1;
    int num_edges = columns.size();
    std::cerr << "num_nodes=" << num_nodes << " | num_edges=" << num_edges << std::endl;

    int* h_person2item = (int *)malloc(sizeof(int) * num_nodes);

    int verbose = 1;

    int elapsed = run_auction(
        num_nodes,
        num_edges,

        h_data,
        h_offsets,
        h_columns,

        h_person2item,

        AUCTION_MAX_EPS,
        AUCTION_MIN_EPS,
        AUCTION_FACTOR,

        NUM_RUNS,
        verbose,
        use_cub
    );
    std::cerr << "elapsed=" << elapsed << std::endl;

    // Print results
    float score = 0;
    for (int i = 0; i < num_nodes; i++) {
        score += h_data[i * num_nodes + h_person2item[i]];
    }

    std::cerr << "score=" << (int)score << std::endl;

    offsets.clear();
    columns.clear();
    data.clear();
}

#endif