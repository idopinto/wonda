## InvBench - Evaluation Dataset

The datasets used in this project are from the [InvBench repository](https://github.com/Anjiang-Wei/InvBench), which contains training and evaluation datasets for loop invariant generation.

- Merged the original easy and hard splits to unified data/orig_programs for re-running by my local verifier for fair baseline comparison.

```
InvBench/
├─ Dataset/
│  ├─ Evaluation/
│  │  ├─ Easy/               # 113 C programs (Easy Split)
│  │  ├─ Hard/               # 113 C programs (Hard Split)
│  │  └─ timing.json         # Evaluation timing data
│  └─ Training/
│     ├─ Programs/           # 3589 C programs for training
│     ├─ invariants.json     # Ground-truth invariants for training programs
│     └─ timing.json         # Training timing data
```

## Issues found in InvBench
Easy - 113
Hard - 113
5 identical files exist in both easy and hard directories and suppose to be only in the easy split (timing < 30sec for all)
1. cohencu-ll_valuebound100_8.c: 25.89s
2. cohencu-ll_valuebound20_8.c: 27.62s
3. egcd-ll_valuebound5_7.c: 26.4s
4. fermat1-ll_2.c: 29.43s
5. geo1-ll_unwindbound1_2.c: 26.88s

9 hard-split files with timing < 30
1. nested_delay_notd2_1.c: 29.34s
2. diamond_1-1_1.c: 26.39s
3. num_conversion_1_1.c: 25.74s
4. ps6-ll_3.c: 27.42s
5. cohencu-ll_valuebound20_9.c: 28.03s
6. rewnifrev2_1.c: 28.33s
7. ps6-ll_unwindbound10_2.c: 27.08s
8. cohencu-ll_valuebound50_8.c: 28.82s
9. egcd-ll_unwindbound5_5.c: 27.49s

One file with no assert (commented out) [filtered out]
1. sll-queue-1_19.c (6.58s, easy)

9 files that has no function definition for __VERIFIER_assert (which causes verification failures) [fixed with normalization]:
1. sll-01-1_8.c
2. sll-01-2_9.c
3. dll-simple-white-blue-2_2.c (__VERIFIER_assert(0);)
4. dll-rb-cnstr_1-2_4.c
5. dll-rb-cnstr_1-2_3.c
6. dll-queue-1_4.c
7. sll-buckets-2_3.c
8. sll-queue-1_13.c
9. sll-queue-1_12.c

1 file with 2 asserts (is it intended?)
1. egcd-ll_valuebound100_5.c

After re-organization:
easy 113 [total] + 9 [easy] - 1 - [no_assert] = 121
hard 113 [total] - 5 [duplicates] - 9 [easy] = 99
New Total = 220
