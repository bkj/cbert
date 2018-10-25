#ifndef __KERNEL_HEADER
#define EMPTY_COL    -99
#define BIG_NEGATIVE -9999999
#define THREADS      1024
#define CUB_BIDDING
#endif

// ----------------------------------------------------------
// Old version

__global__ void __run_bidding_kernel(
    const int num_nodes,
    float *data,
    int *offsets,
    int *columns,
    int *person2item,
    float *bids,
    int *sbids,
    float *prices,
    float *rand,
    float auction_eps
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < num_nodes){
        if(person2item[i] == -1) {

            int start_idx = offsets[i];
            int end_idx   = offsets[i + 1];

            int top1_col;
            float top1_val = -999999;
            float top2_val = -999999;

            int col;
            float tmp_val;

            // Find best zero bid
            for(int col = 0; col < num_nodes; col++) {
                tmp_val = -prices[col];
                if(tmp_val >= top1_val) {
                    if(
                        (tmp_val > top1_val) // ||
                        // (rand[i * num_nodes + col] >= rand[i * num_nodes + top1_col])
                    ) {
                        top2_val = top1_val;
                        top1_col = col;
                        top1_val = tmp_val;
                    }
                } else if(tmp_val > top2_val) {
                    top2_val = tmp_val;
                }
            }

            // Check all nonzero entries first
            for(int idx = start_idx; idx < end_idx; idx++){
                col = columns[idx];
                if(col == EMPTY_COL) {break;}
                tmp_val = data[idx] - prices[col];

                if(tmp_val >= top1_val) {
                    if(
                        (tmp_val > top1_val) // ||
                        // (rand[i * num_nodes + col] >= rand[i * num_nodes + top1_col])
                    ) {
                        top2_val = top1_val;
                        top1_col = col;
                        top1_val = tmp_val;
                    }
                } else if(tmp_val > top2_val) {
                    top2_val = tmp_val;
                }
            }

            // if(i < 16)
            //     printf("k: %d %d:%f\n", i, top1_col, top1_val);

            float bid = top1_val - top2_val + auction_eps;
            bids[num_nodes * top1_col + i] = bid;
            atomicMax(sbids + top1_col, 1);
        }
    }
}

// ----------------------------------------------------------
// Cub code

struct BiddingOp {

    float* prices;
    CUB_RUNTIME_FUNCTION __forceinline__
    BiddingOp(float* prices) : prices(prices) {}

    __device__ __forceinline__
    Entry operator()(const Entry &a, const Entry &b) const {
        int best_row, best_idx;
        bool is_first;
        float best_val, next_best_val, tiebreaker;
        float a_val = a.best_val;
        float b_val = b.best_val;
        if(a.is_first) a_val -= prices[a.idx];
        if(b.is_first) b_val -= prices[b.idx];

        if(
            (a_val > b_val) ||
            ((a.best_val == b.best_val) && (a.idx < b.idx)) // Should (actually) break ties randomly
        ) {
            best_row      = a.row;
            best_idx      = a.idx;
            best_val      = a_val;
            next_best_val = max(a.next_best_val, b_val);
            tiebreaker    = a.tiebreaker;
            is_first      = false;
        } else {
            best_row      = b.row;
            best_idx      = b.idx;
            best_val      = b_val;
            next_best_val = max(a_val, b.next_best_val);
            tiebreaker    = b.tiebreaker;
            is_first      = false;
        }
        return (Entry){best_row, best_idx, best_val, next_best_val, tiebreaker, is_first};
    }
};

struct IsUnassigned
{
    __device__ __forceinline__
    bool operator()(const int &a) const {
        return a == -1;
    }
};


__global__ void __make_entry_array(Entry* out, int* offsets, int* indices, float* data, float* rand_in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        int start = offsets[i];
        int end   = offsets[i + 1];
        for(int offset = start; offset < end; offset++) {
            out[offset] = (Entry){i, indices[offset], data[offset], BIG_NEGATIVE, rand_in[offset], true};
        }
    }
}

__global__ void __fill_price_array(Entry* out, float* in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        out[i] = (Entry){0, 0, in[i], 9999999, 0, false};
    }
}

__global__ void __setFlags(int* out, int* in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        out[i] = (int)(in[i] == -1);
    }
}

__global__ void __scatterBids(float* bids, int* sbids, Entry* in, int num_nodes, int n, float auction_eps) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        float bid = in[i].best_val - in[i].next_best_val + auction_eps;
        // if(in[i].row < 16)
        //     printf("c: %d %d:%f\n", in[i].row, in[i].idx, in[i].best_val);

        bids[num_nodes * in[i].idx + in[i].row] = bid;
        atomicMax(sbids + in[i].idx, 1);
    }
}

void __run_bidding_cub(const int num_nodes, AuctionData ad, float auction_eps) {
    int node_blocks           = 1 + num_nodes / THREADS;
    void   *temp_storage      = NULL;
    size_t temp_storage_bytes = 0;

    // ----------------------------------
    // Find unassigned rows

    __setFlags<<<node_blocks, THREADS>>>(ad.flags, ad.person2item, num_nodes);

    cub::DeviceSelect::Flagged(
        temp_storage, temp_storage_bytes, ad.offsets, ad.flags, ad.unassigned_offsets_start, ad.num_unassigned, num_nodes);
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceSelect::Flagged(
        temp_storage, temp_storage_bytes, ad.offsets, ad.flags, ad.unassigned_offsets_start, ad.num_unassigned, num_nodes);

    cub::DeviceSelect::Flagged(
        temp_storage, temp_storage_bytes, ad.offsets + 1, ad.flags, ad.unassigned_offsets_end, ad.num_unassigned, num_nodes);
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceSelect::Flagged(
        temp_storage, temp_storage_bytes, ad.offsets + 1, ad.flags, ad.unassigned_offsets_end, ad.num_unassigned, num_nodes);

    int* h_num_unassigned = (int*)malloc(1 * sizeof(int));
    cudaMemcpy(h_num_unassigned, ad.num_unassigned, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("c: h_num_unassigned=%d\n", h_num_unassigned[0]);

    // ----------------------------------
    // Run bidding op on unassigned rows

    temp_storage = NULL; temp_storage_bytes = 0;

    BiddingOp bidding_op(ad.prices);
    Entry null_bid = {-1, -1, BIG_NEGATIVE, BIG_NEGATIVE, BIG_NEGATIVE};
    Entry* entry_bid;
    cudaMalloc((void**)&entry_bid, h_num_unassigned[0] * sizeof(Entry));
    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_bytes, ad.entry_array, entry_bid,
        h_num_unassigned[0], ad.unassigned_offsets_start, ad.unassigned_offsets_end, bidding_op, null_bid);
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_bytes, ad.entry_array, entry_bid,
        h_num_unassigned[0], ad.unassigned_offsets_start, ad.unassigned_offsets_end, bidding_op, null_bid);

    // ----------------------------------
    // Broadcast bids to bids

    int tmp_blocks = 1 + h_num_unassigned[0] / THREADS;
    __scatterBids<<<tmp_blocks, THREADS>>>(ad.bids, ad.sbids, entry_bid, num_nodes, h_num_unassigned[0], auction_eps);

    cudaDeviceSynchronize();
    // printf("--\n");
}

// ----------------------------------------------------------
// Wrapper

void run_bidding(const int num_nodes, AuctionData ad, float auction_eps, bool cub) {
    if(cub) {
        __run_bidding_cub(num_nodes, ad, auction_eps);
    } else {
        int node_blocks = 1 + num_nodes / THREADS;
        __run_bidding_kernel<<<node_blocks, THREADS>>>(
            num_nodes,
            ad.data,
            ad.offsets,
            ad.columns,
            ad.person2item,
            ad.bids,
            ad.sbids,
            ad.prices,
            ad.rand,
            auction_eps
        );
    }
}