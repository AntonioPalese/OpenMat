

#include <cooperative_groups.h>


__global__  void kernel()
{
    //namespace cp = cooperative_groups;
    int tid = cooperative_groups::this_grid().thread_rank();
    cooperative_groups::__v1::cluster_group cluster = cooperative_groups::__v1::this_cluster();

    //cg::cluster_group cluster = cg::this_cluster();
}

int main()
{

}


