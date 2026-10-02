# Cross-check against the dissertation (Table 1)

Raw aggregate rates from `data/processed/regional_monthly.csv` vs the dissertation's adjusted rate ratios. A sanity check on direction and ordering, not a replication.

## Region (relative to London)

| region                   |   this_project |   dissertation_RR |   rank_this |   rank_diss |
|:-------------------------|---------------:|------------------:|------------:|------------:|
| East of England          |          1.81  |             1.668 |           2 |           3 |
| London                   |          1     |             1     |           7 |           7 |
| Midlands                 |          1.635 |             1.48  |           5 |           6 |
| North East and Yorkshire |          1.765 |             1.794 |           4 |           2 |
| North West               |          1.588 |             1.499 |           6 |           5 |
| South East               |          1.789 |             1.51  |           3 |           4 |
| South West               |          2.065 |             1.856 |           1 |           1 |

Spearman rank correlation: 0.86

## Calendar month (relative to January)

|    |   this_project |   dissertation_RR |   years_averaged |
|---:|---------------:|------------------:|-----------------:|
|  1 |          1     |             1     |                2 |
|  2 |          1.03  |             1.024 |                2 |
|  3 |          1.083 |             1.076 |                2 |
|  4 |          1.177 |             1.165 |                2 |
|  5 |          1.134 |             1.118 |                2 |
|  6 |          1.133 |             1.106 |                2 |
|  7 |          1.089 |             1.064 |                2 |
|  8 |          1.126 |             1.081 |                2 |
|  9 |          1.272 |             1.209 |                2 |
| 10 |          1.469 |             1.361 |                3 |
| 11 |          1.174 |             1.123 |                2 |
| 12 |          0.999 |             1.015 |                2 |

Pearson r = 0.99, Spearman = 0.99
