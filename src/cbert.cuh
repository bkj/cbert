#ifndef __AUCTION_HEADER
#define __AUCTION_HEADER

#define THREADS 1024

typedef struct {
    int row;
    int idx;
    float best_val;
    float next_best_val;
    float tiebreaker;
    bool is_first;
} Entry;

struct AuctionData {

    // Data
    float *data;
    int   *columns;
    int   *offsets;

    // Auction data structures
    int   *num_assigned;
    int   *person2item;
    int   *item2person;
    float *prices;
    int   *sbids;
    float *bids;
    float *rand;

    // Cub auction data structures
    Entry* entry_array;
    int* flags;
    int* num_unassigned;
    int* unassigned_offsets_start;
    int* unassigned_offsets_end;
} ;

#endif