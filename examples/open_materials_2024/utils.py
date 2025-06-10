import heapq, random


def balance_load(dataset_list, nranks, me):
    indexed_lists = []
    for i, lstitem in enumerate(dataset_list):
        indexed_lists.append({"id": i, "num_samples": lstitem["num_samples"]})

    # Create a sorted list in desc order that records the id and count of num_samples
    sorted_l = sorted(indexed_lists, key=lambda x: x["num_samples"], reverse=True)

    # Now create a heap
    heap = []
    for i in range(nranks):
        heapq.heappush(heap, (0, i, []))  # load, rank, assigned datasets

    # Start assignment
    for lstinfo in sorted_l:
        idx = lstinfo["id"]
        num_samples_count = lstinfo["num_samples"]

        load, rank, assigned_datasets = heapq.heappop(heap)
        assigned_datasets.append(dataset_list[idx])
        load += num_samples_count
        heapq.heappush(heap, (load, rank, assigned_datasets))

    # Final printing
    for i in range(nranks):
        load, rank, assigned_datasets = heapq.heappop(heap)
        if rank == me:
            print(
                f"load balancing. Rank {rank}: num_samples: {load}, number of datasets: {len(assigned_datasets)}"
            )
            return assigned_datasets


def main():
    # Create mock lists of datasets with samples
    ndatasets = 100
    dataset_list = []
    total_num_samples = 0
    for i in range(ndatasets):
        num_samples = random.randint(100, 1000)
        total_num_samples += num_samples
        dataset = []
        for j in range(num_samples):
            dataset.append(j)
        dataset_list.append(dataset)

    # print list of datasets
    print(f"total number of samples: {total_num_samples}. dataset list sizes")
    for dataset in dataset_list:
        print(f"{len(dataset)}")

    # distribute workload
    balance_load(dataset_list, 7)


if __name__ == "__main__":
    main()
