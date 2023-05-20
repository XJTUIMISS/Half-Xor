#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <iomanip>
#include <math.h>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <atomic>
#include "MurmurHash3.h"

using namespace std;
using namespace chrono;

unsigned hll_fd_seed = 128723;
unsigned hll_pro_seed = 128723;
unsigned long long database_rows;

void init_hll_pro_seed()
{
	hll_pro_seed = rand();
}

void feistel_encrypt(unsigned &L, unsigned &R)
{
	unsigned long long rand_unsigned1 = 1543454366035403, rand_unsigned2 = 240823032534547601, re;
	L = L ^ unsigned(R + rand_unsigned1);
	R = R ^ unsigned(L + rand_unsigned2);
}

unsigned long long generate_dataset(unsigned long long * dataset, unsigned long long n)
{
	// Here I first generate n distinct random 64-bit numbers using feistel network.
	unsigned long long i, j, rows = 0;
	unsigned L, R, temp_duplicates;
	for (i = 0; i < n; ++i)
	{
		L = i >> 32;
		R = i;
		feistel_encrypt(L, R);
		temp_duplicates = (rand() % 10) + 1;
		for (j = 0; j < temp_duplicates; ++j)
		{
			dataset[rows] = L * 4294967296 + R;
			rows += 1;
		}
	}
	return rows;
}

int rank_bits(unsigned long long bits, int w)
{
	int i = 0;
	unsigned long long musk = 1;
	for (i = 0; i < w - 1; ++i)
	{
		if ((musk & bits) != 0)
		{
			return i;
		}
		bits = bits >> 1;
	}
	return 31;
}

unsigned * hll_pro_generate_sketch(int m, unsigned long long n, int w, unsigned long long * dataset)
{
	unsigned * hll_pro_sketch = new unsigned[m * w]();
	unsigned long long i, j;
	int index_j = 0, m_bits = 0, index_i = 0, temp;
	m_bits = log2(m);
	unsigned long long hash_outcome[2] = {0};
	for (i = 0; i < database_rows; ++i)
	{
		MurmurHash3_x64_128(&dataset[i], 8, hll_pro_seed, hash_outcome);
		index_j = rank_bits(hash_outcome[0], w);
		index_i = (hash_outcome[1] >> (64 - m_bits));
		temp = rand();
		MurmurHash3_x64_128(&temp, 4, hll_pro_seed, hash_outcome);
		if (hll_pro_sketch[index_i * w + index_j] <= 128)
		{
			hll_pro_sketch[index_i * w + index_j] += 1;
		}
		if (rank_bits(hash_outcome[0], 128) == (hll_pro_sketch[index_i * w + index_j] - 129))
		{
			hll_pro_sketch[index_i * w + index_j] += 1;
		}
	}
	return hll_pro_sketch;
}

double get_alpha_m(int m)
{
	if (m == 16) return 0.673;
	if (m == 32) return 0.697;
	if (m == 64) return 0.709;
	if (m >= 128) return (0.7213 * double(m) / (double(m) + 1.079));
	return 0.673;
}

double hll_pro_estimate(int m, int w, unsigned * hll_pro_sketch)
{
	int i = 0, j = 0, flag = 0, v = 0;
	double estimate_1, estimate_2, z = 0;
	for (i = 0; i < m; ++i)
	{
		flag = 0;
		for (j = w - 1; j >= 0; --j)
		{
			if (hll_pro_sketch[i * w + j] > 0)
			{
				z += pow(0.5, j + 1);
				flag = 1;
				break;
			}
		}
		if (flag == 0)
		{
			z += 1.0;
			++v;
		}
	}
	estimate_1 = get_alpha_m(m) * double(m) * double(m) / z;
	if (estimate_1 > (2.5 * double(m)))
	{
		return estimate_1;
	}
	z = double(v) / double(m);
	estimate_2 = -1 * double(m) * log(z);
	return estimate_2;
}

int main()
{
	unsigned long long n = 10000, i;
	int m = 256, w = 32;
	double estimate = 0;
	srand(374345);
	init_hll_pro_seed();
	unsigned long long * dataset = new unsigned long long[10*n];
	database_rows = generate_dataset(dataset, n);
	unsigned * hll_pro_sketch = hll_pro_generate_sketch(m, n, w, dataset);
	estimate = hll_pro_estimate(m, w, hll_pro_sketch);
	cout << endl << setw(35) << "real_NDV:" << setw(20) << n << endl;
	cout << setw(35) << "storage:" << setw(20) << "8KB" << endl;
	cout << setw(35) << "w:" << setw(20) << w << endl << endl;
	cout << setw(35) << "HLL-Pro" << setw(20) << estimate << endl << endl;
	delete[] hll_pro_sketch;
	delete[] dataset;
	return 0;
}