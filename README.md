## Half-Xor: A Fully-Dynamic Sketch for Estimating the Number of Distinct Values in Big Tables
![gcc](https://img.shields.io/badge/gcc-8.1.0-yellow) ![mmh3](https://img.shields.io/badge/mmh3-3.0.0-blue)


This repository includes our realization of Half-Xor, which can be used to estimate the number of distinct values (i.e. NDV) in a column of big tables when both insert and delete happens. Though several previous sketch methods (e.g. [HyperLogLog Sketch](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)) have been proposed, they cannot or are costly to manage fully-dynamic scenarios where data is frequently inserted into and deleted from the table. Our Half-Xor sketch consists of a compact bit matrix and a small counter array, and it needs to set a few bits and update a counter when handling a data insertion/deletion. Compared with the mergeable state-of-the-art method, our experimental results demonstrate that our method Half-Xor is up to 6.6 times more accurate under the same memory usage and reduces the memory usage by up to 16 times to achieve the same estimation accuracy.

### Datasets

We involve synthetic datasets, real world datasets and TPCH, TPCDS benchmarks in our experiments. For synthetic datasets, we first generate 𝑛 distinct integers drawn from 64-bit integer space without replacement. Then for every one of these 𝑛 elements, we duplicate it 𝑘 times, where 𝑘 is a random variable chosen from {1, 2, 3, . . . , 10} uniformly at random. Then we insert all the duplicated data values into column 𝑎 of table 𝑇. Besides, all the primary key column values in table 𝑇 are sampled from 64-bit integer space without replacement. 

For real-world datasets, we use the databases from the relational database repository, which can be accessed through

```url
https://relational.fit.cvut.cz/search
```

Besides, more details about TPCH and TPCDS databases can be found through this

```
https://www.tpc.org/default5.asp
```

 Details of real world datasets and TPCH, TPCDS datasets are listed below:

|  Database  | number tables | number of columns | minimal NDV | maximal NDV |  Size   |
| :--------: | :-----------: | :---------------: | :---------: | :---------: | :-----: |
| Accidents  |       2       |        26         |      2      |   229181    | 234.5MB |
|   Chess    |       2       |        45         |      1      |     633     |  300KB  |
|  Consumer  |       2       |        18         |      2      |   370640    | 337.6MB |
|   Credit   |       6       |        23         |      7      |    45866    | 317.9MB |
|  TPCH-10   |       8       |        61         |      1      |  34378900   |  10GB   |
|  TPCDS-10  |      24       |        425        |      1      |   1702210   |  10GB   |
| TPCDS-100  |      24       |        425        |      1      |  16966500   |  100GB  |
| TPCDS-1000 |      24       |        425        |      1      |  169676054  | 1000GB  |




### Methods implemented

|Method            |Data Structure                       |Description                             |Reference|
|:------------------:|:-----------------------------------:|:---------------------------------------:|:----------------------:|
|HyperLogLog-FD        |counter matrix                            |large error and large memory cost                 |[HLL-FD.cpp](HLL-FD.cpp)      |
|HyperLogLog-Pro|probabilistic counter matrix|cannot merge and unstable performance|[HLL-Pro.cpp](HLL-Pro.cpp)|
|Half-Xor |bit matrix and counter array|original half-xor|[Half-Xor.cpp](Half-Xor.cpp)      |
|Half-Xor (order statistics)|bit matrix and counter array|acceleration of original half-xor|[Half-Xor.cpp](Half-Xor.cpp)      |
|Half-Xor (stochastic averaging)|bit matrix and counter array|simulation and acceleration of original half-xor|[Half-Xor.cpp](Half-Xor.cpp)      |
|Half-Xor (bucket hashing)|bit matrix and counter array|simulation and acceleration of original half-xor, most efficeint|[Half-Xor.cpp](Half-Xor.cpp)|



To compile the codes, use:

```shell
make hll-fd
make hll-pro
make half-xor
```

Besides, we implement the three estimates based on Half-Xor. The Half-Xor-EZ, Half-Xor-IVW are in Half-Xor.cpp and Half-Xor-ML in Half-Xor-ML/*.
