
__global__ void __run_assignment_kernel(
    const int num_nodes,
    int *person2item,
    int *item2person,
    float *bids,
    int *sbids,
    float *prices,
    int *num_assigned
)
{

    int j = blockDim.x * blockIdx.x + threadIdx.x; // item index
    if(j < num_nodes) {
        if(sbids[j] != 0) {
            float high_bid  = -1;
            int high_bidder = -1;

            float tmp_bid;
            for(int i = 0; i < num_nodes; i++){
                tmp_bid = bids[num_nodes * j + i];
                if(tmp_bid > high_bid){
                    high_bid    = tmp_bid;
                    high_bidder = i;
                }
            }

            int current_person = item2person[j];
            if(current_person != -1){
                person2item[current_person] = -1;
            } else {
                atomicAdd(num_assigned, 1);
            }

            prices[j]                += high_bid;
            person2item[high_bidder] = j;
            item2person[j]           = high_bidder;
        }
    }
}

void __run_assignment_cub(const int num_nodes, AuctionData ad) {

    // void   *temp_storage      = NULL;
    // size_t temp_storage_bytes = 0;

    // int* h_offsets = (int*)malloc((num_nodes + 1) * sizeof(int));
    // for(int i = 0; i < num_nodes + 1; i++) {
    //     h_offsets[i] = i * num_nodes;
    // }

    // int* d_offsets;
    // cudaMalloc((void**)&d_offsets, (num_nodes + 1) * sizeof(int));
    // cudaMemcpy(d_offsets, h_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_bytes, ad.entry_array, entry_bid,
    //     h_num_unassigned[0], ad.unassigned_offsets_start, ad.unassigned_offsets_end, bidding_op, null_bid);
}


void run_assignment(const int num_nodes, AuctionData ad, bool cub) {
    if(cub) {
        __run_assignment_cub(num_nodes, ad);
    } else {
        int node_blocks = 1 + num_nodes / THREADS;
        __run_assignment_kernel<<<node_blocks, THREADS>>>(
            num_nodes,
            ad.person2item,
            ad.item2person,
            ad.bids,
            ad.sbids,
            ad.prices,
            ad.num_assigned
       );
    }
}