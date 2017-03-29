# Report 

## Results

**N** = 100

| n | Serial | CUDA w/o SharedMem | Acceleration | CheckResult | CUDA w/ SharedMem | Acceleration | CheckResult |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 |  0,006358 | 0,000090 | 70,644444 | Correct | 0,000068 | 1,323529 | Correct |
| 1 |  0,001210 | 0,000085 | 14,235294 | Correct | 0,000069 | 1,231884 | Correct |
| 2 |  0,002542 | 0,000089 | 28,561798 | Correct | 0,000066 | 1,348485 | Correct |
| 3 |  0,006377 | 0,000089 | 71,651685 | Correct | 0,000066 | 1,348485 | Correct |
| 4 |  0,001227 | 0,000089 | 13,786517 | Correct | 0,000064 | 1,390625 | Correct |
| 5 |  0,006322 | 0,000086 | 73,511628 | Correct | 0,000064 | 1,343750 | Correct |
| 6 |  0,003165 | 0,000085 | 37,235294 | Correct | 0,000065 | 1,307692 | Correct |
| 7 |  0,003171 | 0,000090 | 35,233333 | Correct | 0,000064 | 1,406250 | Correct |
| 8 |  0,004223 | 0,000086 | 49,104651 | Correct | 0,000064 | 1,343750 | Correct |
| 9 |  0,006349 | 0,000085 | 74,694118 | Correct | 0,000064 | 1,328125 | Correct |
| 10 |  0,001203 | 0,000086 | 13,988372 | Correct | 0,000063 | 1,365079 | Correct |
| 11 |  0,001220 | 0,000085 | 14,352941 | Correct | 0,000063 | 1,349206 | Correct |
| 12 |  0,001202 | 0,000087 | 13,816092 | Correct | 0,000066 | 1,318182 | Correct |
| 13 |  0,001216 | 0,000085 | 14,305882 | Correct | 0,000064 | 1,328125 | Correct |
| 14 |  0,001209 | 0,000086 | 14,058140 | Correct | 0,000063 | 1,365079 | Correct |
| 15 |  0,001201 | 0,000088 | 13,647727 | Correct | 0,000064 | 1,375000 | Correct |
| 16 |  0,001203 | 0,000084 | 14,321429 | Correct | 0,000064 | 1,312500 | Correct |
| 17 |  0,001210 | 0,000089 | 13,595506 | Correct | 0,000063 | 1,412698 | Correct |
| 18 |  0,001208 | 0,000088 | 13,727273 | Correct | 0,000064 | 1,375000 | Correct |
| 19 |  0,001204 | 0,000086 | 14,000000 | Correct | 0,000065 | 1,323077 | Correct |

**N** = 500

| n | Serial | CUDA w/o SharedMem | Acceleration | CheckResult | CUDA w/ SharedMem | Acceleration | CheckResult |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 |  0,149650 | 0,002552 | 58,640282 | Correct | 0,001115 | 2,288789 | Correct |
| 1 |  0,175268 | 0,002571 | 68,171140 | Correct | 0,001126 | 2,283304 | Correct |
| 2 |  0,148949 | 0,002525 | 58,989703 | Correct | 0,001114 | 2,266607 | Correct |
| 3 |  0,178665 | 0,002571 | 69,492415 | Correct | 0,001113 | 2,309973 | Correct |
| 4 |  0,172430 | 0,002589 | 66,601004 | Correct | 0,001135 | 2,281057 | Correct |
| 5 |  0,148897 | 0,002562 | 58,117486 | Correct | 0,001112 | 2,303957 | Correct |
| 6 |  0,149601 | 0,002577 | 58,052386 | Correct | 0,001132 | 2,276502 | Correct |
| 7 |  0,148787 | 0,002540 | 58,577559 | Correct | 0,001140 | 2,228070 | Correct |
| 8 |  0,166940 | 0,002546 | 65,569521 | Correct | 0,001108 | 2,297834 | Correct |
| 9 |  0,148790 | 0,002521 | 59,020230 | Correct | 0,001113 | 2,265049 | Correct |
| 10 |  0,149941 | 0,002554 | 58,708301 | Correct | 0,001133 | 2,254192 | Correct |
| 11 |  0,148793 | 0,002529 | 58,834717 | Correct | 0,001104 | 2,290761 | Correct |
| 12 |  0,149170 | 0,002552 | 58,452194 | Correct | 0,001111 | 2,297030 | Correct |
| 13 |  0,148780 | 0,002576 | 57,756211 | Correct | 0,001117 | 2,306177 | Correct |
| 14 |  0,148803 | 0,002589 | 57,475087 | Correct | 0,001129 | 2,293180 | Correct |
| 15 |  0,175270 | 0,002579 | 67,960450 | Correct | 0,001109 | 2,325518 | Correct |
| 16 |  0,148932 | 0,002535 | 58,750296 | Correct | 0,001114 | 2,275583 | Correct |
| 17 |  0,175245 | 0,002587 | 67,740626 | Correct | 0,001123 | 2,303651 | Correct |
| 18 |  0,171669 | 0,002554 | 67,215740 | Correct | 0,001124 | 2,272242 | Correct |
| 19 |  0,169803 | 0,002563 | 66,251658 | Correct | 0,001115 | 2,298655 | Correct |

**N** = 1000

| n | Serial | CUDA w/o SharedMem | Acceleration | CheckResult | CUDA w/ SharedMem | Acceleration | CheckResult |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 |  1,195667 | 0,013959 | 85,655634 | Correct | 0,005229 | 2,669535 | Correct |
| 1 |  1,193665 | 0,013826 | 86,334804 | Correct | 0,005219 | 2,649167 | Correct |
| 2 |  1,195836 | 0,013750 | 86,969891 | Correct | 0,005228 | 2,630069 | Correct |
| 3 |  1,192630 | 0,013861 | 86,042133 | Correct | 0,005597 | 2,476505 | Correct |
| 4 |  1,218172 | 0,013798 | 88,286128 | Correct | 0,005230 | 2,638241 | Correct |
| 5 |  1,193084 | 0,013762 | 86,694085 | Correct | 0,005260 | 2,616350 | Correct |
| 6 |  1,214631 | 0,013846 | 87,724325 | Correct | 0,005256 | 2,634323 | Correct |
| 7 |  1,193153 | 0,013836 | 86,235400 | Correct | 0,005602 | 2,469832 | Correct |
| 8 |  1,194667 | 0,013996 | 85,357745 | Correct | 0,005224 | 2,679173 | Correct |
| 9 |  1,213203 | 0,013828 | 87,735247 | Correct | 0,005236 | 2,640947 | Correct |
| 10 |  1,211807 | 0,014110 | 85,882849 | Correct | 0,005228 | 2,698929 | Correct |
| 11 |  1,211548 | 0,013779 | 87,927135 | Correct | 0,005243 | 2,628076 | Correct |
| 12 |  1,194283 | 0,013804 | 86,517169 | Correct | 0,005256 | 2,626332 | Correct |
| 13 |  1,196635 | 0,013735 | 87,123043 | Correct | 0,005235 | 2,623687 | Correct |
| 14 |  1,217863 | 0,013765 | 88,475336 | Correct | 0,005544 | 2,482864 | Correct |
| 15 |  1,195459 | 0,014295 | 83,627772 | Correct | 0,005242 | 2,727013 | Correct |
| 16 |  1,219542 | 0,013761 | 88,623065 | Correct | 0,005223 | 2,634693 | Correct |
| 17 |  1,217003 | 0,013802 | 88,175844 | Correct | 0,005247 | 2,630455 | Correct |
| 18 |  1,193827 | 0,013770 | 86,697676 | Correct | 0,005225 | 2,635407 | Correct |
| 19 |  1,220131 | 0,013746 | 88,762622 | Correct | 0,005562 | 2,471413 | Correct |

**N** = 1500

| n | Serial | CUDA w/o SharedMem | Acceleration | CheckResult | CUDA w/ SharedMem | Acceleration | CheckResult |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 |  5,108654 | 0,046234 | 110,495609 | Correct | 0,014417 | 3,206909 | Correct |
| 1 |  5,043349 | 0,046842 | 107,667243 | Correct | 0,014370 | 3,259708 | Correct |
| 2 |  5,092390 | 0,046474 | 109,575031 | Correct | 0,014354 | 3,237704 | Correct |
| 3 |  5,017608 | 0,046853 | 107,092566 | Correct | 0,014097 | 3,323615 | Correct |
| 4 |  5,082576 | 0,046805 | 108,590450 | Correct | 0,014048 | 3,331791 | Correct |
| 5 |  5,074446 | 0,046475 | 109,186573 | Correct | 0,014397 | 3,228103 | Correct |
| 6 |  4,936007 | 0,046800 | 105,470235 | Correct | 0,014026 | 3,336660 | Correct |
| 7 |  4,936221 | 0,046802 | 105,470300 | Correct | 0,014435 | 3,242258 | Correct |
| 8 |  5,123099 | 0,046662 | 109,791672 | Correct | 0,014344 | 3,253067 | Correct |
| 9 |  4,939994 | 0,046858 | 105,424773 | Correct | 0,014049 | 3,335326 | Correct |
| 10 |  4,949172 | 0,046898 | 105,530556 | Correct | 0,014450 | 3,245536 | Correct |
| 11 |  4,969428 | 0,046831 | 106,114070 | Correct | 0,013976 | 3,350816 | Correct |
| 12 |  5,038730 | 0,046486 | 108,392419 | Correct | 0,014466 | 3,213466 | Correct |
| 13 |  5,087221 | 0,046876 | 108,525066 | Correct | 0,014411 | 3,252793 | Correct |
| 14 |  5,081982 | 0,046870 | 108,427182 | Correct | 0,014042 | 3,337844 | Correct |
| 15 |  5,115071 | 0,046463 | 110,089125 | Correct | 0,014392 | 3,228391 | Correct |
| 16 |  4,985260 | 0,046771 | 106,588698 | Correct | 0,014369 | 3,254993 | Correct |
| 17 |  4,999516 | 0,046996 | 106,381735 | Correct | 0,014467 | 3,248497 | Correct |
| 18 |  5,093095 | 0,046922 | 108,543860 | Correct | 0,014405 | 3,257341 | Correct |
| 19 |  4,975898 | 0,046820 | 106,277189 | Correct | 0,014048 | 3,332859 | Correct |

## Results on charts

* Chart of resulted times: Comparison between Serial, CUDA with and without shared memory.

![Image 1](https://github.com/jimyandres/HPC/blob/master/CUDA/matrixMult_V2_tiling/Report/Time_CUDA_Vs_Serial.png)

* Chart of resulted times: Comparison between CUDA without shared memory and with shared memory.

![Image 2](https://github.com/jimyandres/HPC/blob/master/CUDA/matrixMult_V2_tiling/Report/Time_CUDA_comparison.png)

* Chart of resulted accelerations: Comparison between CUDA without shared memory and with shared memory.

![Image 3](https://github.com/jimyandres/HPC/blob/master/CUDA/matrixMult_V2_tiling/Report/Acc_CUDA_comparison.png)

* Chart of resulted acceleration: CUDA without shared memory.

![Image 4](https://github.com/jimyandres/HPC/blob/master/CUDA/matrixMult_V2_tiling/Report/Acc_CUDA_without_SM.png)