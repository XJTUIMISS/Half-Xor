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

unsigned half_xor_seed_array_d;
unsigned * half_xor_seed_uniform;
unsigned * half_xor_seed_xor;
unsigned * half_xor_seed_rank;
unsigned * half_xor_sketch;
unsigned long long database_rows;

void init_half_xor_seed(int m)
{
	half_xor_seed_uniform = new unsigned[m];
	half_xor_seed_xor = new unsigned[m];
	half_xor_seed_rank = new unsigned[m];
	int i = 0;
	for (i = 0; i < m; ++i)
	{
		half_xor_seed_uniform[i] = rand();
		half_xor_seed_xor[i] = rand();
		half_xor_seed_rank[i] = rand();
	}
	half_xor_seed_array_d = rand();
}

void feistel_encrypt(unsigned &L, unsigned &R)
{
	unsigned long long rand_unsigned1 = 1543454366035403, rand_unsigned2 = 240823032534547601, re;
	L = L ^ unsigned(R + rand_unsigned1);
	R = R ^ unsigned(L + rand_unsigned2);
}

unsigned long long generate_dataset(unsigned long long * data_values, unsigned long long * primary_keys, unsigned long long n)
{
	// Here I generate n distinct random 64-bit numbers using feistel network. Each data value is duplicated 1 to 10 times.
	unsigned long long i, j, rows = 0, primarykey = 1000000000000;
	unsigned L, R, PL, PR, temp_duplicates;
	for (i = 0; i < n; ++i)
	{
		L = i >> 32;
		R = i;
		feistel_encrypt(L, R);
		temp_duplicates = (rand() % 10) + 1;
		for (j = 0; j < temp_duplicates; ++j)
		{
			data_values[rows] = L * 4294967296 + R;
			primarykey++;
			PL = (primarykey >> 32);
			PR = primarykey;
			feistel_encrypt(PL, PR);
			primary_keys[rows] = PL * 4294967296 + PR;
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

int poisson(double u, double lambda)
{
	int x = 0, i = 0;
	double p = exp(-lambda);
	double s = p;
	for (i = 0; i < 100; ++i)
	{
		if (u <= s)
		{
			break;
		}
		x += 1;
		p = p * lambda / double(x);
		s += p;
	}
	return x;
}

double phi_fun(int n, int w, double lam)
{
	int i = 0;
	double z = w / 2;
	for (i = 0; i < w - 1; ++i)
	{
		z -= exp(-double(n) * lam / pow(2.0, double(i + 1))) / 2;
	}
	z -= exp(-double(n) * lam / pow(2.0, double(w - 1))) / 2;
	return z;
}

double phi_fun_derived(int n, int w, double lam)
{
	int i = 0;
	double z = 0;
	for (i = 0; i < w - 1; ++i)
	{
		z += lam * exp(-double(n) * lam / pow(2.0, double(i + 1))) / pow(2.0, i + 2);
	}
	z += lam * exp(-double(n) * lam / pow(2.0, double(w - 1))) / pow(2.0, w);
	return z;
}

double phi_fun_estimator(int m, int w, double lam)
{
	int i = 0, count = 0;
	double z = 0, n = 10;
	for (i = 0; i < m * w; ++i)
	{
		count += half_xor_sketch[i];
	}
	z = double(count) / double(m);
	while((phi_fun(n, w, lam)-z) * (phi_fun(n+1, w, lam)-z) > 0)
	{
		n = n - (phi_fun(n, w, lam)-z) / phi_fun_derived(n, w, lam);
	}
	return n;
}

double EZ_estimate(int m, int w, double lam)
{
	double n_1 = phi_fun_estimator(m, w, lam), n_2, v = 0;
	int i = 0, emp = 0;
	if (n_1 > double(3 * m))
	{
		return n_1;
	}
	for (i = 0; i < m; ++i)
	{
		if (half_xor_sketch[m * w + i] == 0) emp++;
	}
	v = double(emp) / m;
	if (v == 0) return -1.0;
	n_2 = -1 * double(m) * log(v);
	return n_2;
}

double IVW_estimate(int m, int w, double lam, double esti)
{
	// here we assume w = 32
	double n[32] = {0}, n_d = 0;
	double var[32] = {0};
	double p = 0, n_f = 0, denomi = 0, var_d = 0, var_b = 0;
	int i = 0, j = 0, z = 0, emp = 0;
	for (i = 0; i < w; ++i)
	{
		z = 0;
		for (j = 0; j < m; ++j)
		{
			if(half_xor_sketch[j * w + i] == 1)
			{
				z++;
			}
		}
		if (i < (w - 1)) {p = 1.0 / pow(2.0, i + 1);} else {p = 1.0 / pow(2.0, i);}
		if (z < int(0.5 * m))
		{
			n[i] = -1 * log(1.0 - 2.0 * double(z) / double(m)) / (lam * p);
			var[i] = (1 - exp(-2.0 * esti * lam * p)) / (double(m) * pow(lam * p * exp(-1.0 * esti * lam * p), 2.0));
			denomi += 1.0 / var[i];
		}
		else
		{
			n[i] = -1.0;
			var[i] = -1.0;
		}
	}
	for (i = 0; i < m; ++i)
	{
		if (half_xor_sketch[m * w + i] == 0) emp++;
	}
	if (emp == 0)
	{
		n_d = -1.0;
	}
	else
	{
		n_d = -1.0 * double(m) * log(double(emp) / double(m));
	}
	for (i = 0; i < w; ++i)
	{
		if (n[i] >= 0)
		{
			n_f += (n[i] / var[i]) / denomi;
		}
	}
	var_d = double(m) * (exp(n_f/m) - n_f/m - 1);
	var_b = 1.0 / denomi;
	if (var_b > var_d)
	{
		n_f = n_d;
	}
	return n_f;
}

void half_xor_generate_sketch(int m, unsigned long long n, int w, double lambda, unsigned long long * data_values, unsigned long long * primary_keys)
{
	half_xor_sketch = new unsigned[m * w + m]();
	unsigned long long i, j, k, xor_rank = 0, r_base = 10000000000, x;
	unsigned long long temp_pair_1[2], temp_pair_2[2];
	double i_64 = pow(2, 64);
	int index_j = 0, m_bits = 0, index_i = 0, index_d_array = 0;
	unsigned long long hash_uniform_outcom[2] = {0};
	unsigned long long hash_xor_outcom[2] = {0};
	unsigned long long hash_rank_outcom[2] = {0};
	m_bits = log2(m);
	for (i = 0; i < database_rows; ++i)
	{
		MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_array_d, hash_uniform_outcom);
		index_d_array = (hash_uniform_outcom[0] >> (64 - m_bits));
		half_xor_sketch[m * w + index_d_array] += 1;
		for (j = 0; j < m; ++j)
		{
			MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_uniform[j], hash_uniform_outcom);
			x = poisson(double(hash_uniform_outcom[0]) / i_64, lambda);
			for (k = 1; k <= x; ++k)
			{
				temp_pair_1[0] = data_values[i];
				temp_pair_1[1] = k;
				MurmurHash3_x64_128(temp_pair_1, 16, half_xor_seed_rank[j], hash_rank_outcom);
				xor_rank = rank_bits(hash_rank_outcom[0], w);
				temp_pair_2[0] = primary_keys[i];
				temp_pair_2[1] = (j << 48) + k;
				MurmurHash3_x64_128(temp_pair_2, 16, half_xor_seed_xor[j], hash_xor_outcom);
				half_xor_sketch[j * w + xor_rank] ^= (hash_xor_outcom[0] >> 63);
			}
		}
	}
}

void half_xor_generate_sketch_order_statistics(int m, unsigned long long n, int w, double lambda, unsigned long long * data_values, unsigned long long * primary_keys)
{
	half_xor_sketch = new unsigned[m * w + m]();
	unsigned * permu = new unsigned[m]();
	unsigned * permu_origin = new unsigned[m]();
	unsigned long long i, j, k, xor_rank = 0, r_base = 10000000000, x;
	unsigned long long temp_pair_1[2], temp_pair_2[2];
	double i_64 = pow(2, 64), uniform_variable = 1.0;
	int index_j = 0, m_bits = 0, index_i = 0, index_d_array = 0, temp_y, temp;
	unsigned long long hash_uniform_outcom[2] = {0};
	unsigned long long hash_xor_outcom[2] = {0};
	unsigned long long hash_rank_outcom[2] = {0};
	m_bits = log2(m);
	for (j = 0; j < m; ++j)
	{
		permu_origin[j] = j;
	}
	for (i = 0; i < database_rows; ++i)
	{
		uniform_variable = 1.0;
		MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_array_d, hash_uniform_outcom);
		index_d_array = (hash_uniform_outcom[0] >> (64 - m_bits));
		half_xor_sketch[m * w + index_d_array] += 1;
		memcpy(permu, permu_origin, 4 * m);
		for (j = m - 1; j >= 0; --j)
		{
			MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_uniform[j], hash_uniform_outcom);
			uniform_variable = uniform_variable * pow(double(hash_uniform_outcom[0]) / i_64, 1.0 / (j + 1));
			x = poisson(uniform_variable, lambda);
			if (x == 0)
			{
				break;
			}
			temp_y = floor((double(hash_uniform_outcom[1]) / i_64) * (j + 1));
			if (temp_y >= (j + 1)) {temp_y = j;} else if (temp_y < 0) {temp_y = 0;}
			{temp = permu[temp_y]; permu[temp_y] = permu[j]; permu[j] = temp;}
			for (k = 1; k <= x; ++k)
			{
				temp_pair_1[0] = data_values[i];
				temp_pair_1[1] = k;
				MurmurHash3_x64_128(temp_pair_1, 16, half_xor_seed_rank[permu[j]], hash_rank_outcom);
				xor_rank = rank_bits(hash_rank_outcom[0], w);
				temp_pair_2[0] = primary_keys[i];
				temp_pair_2[1] = (k << 48) + permu[j];
				MurmurHash3_x64_128(temp_pair_2, 16, half_xor_seed_xor[permu[j]], hash_xor_outcom);
				half_xor_sketch[permu[j] * w + xor_rank] ^= (hash_xor_outcom[0] >> 63);
			}
		}
	}
	delete[] permu;
	delete[] permu_origin;
}

void half_xor_generate_sketch_order_statistics_insert_and_delete(double ratio, int m, unsigned long long n, int w, double lambda, unsigned long long * data_values, unsigned long long * primary_keys)
{
	half_xor_sketch = new unsigned[m * w + m]();
	unsigned * permu = new unsigned[m]();
	unsigned * permu_origin = new unsigned[m]();
	unsigned long long i, j, k, xor_rank = 0, r_base = 10000000000, x;
	unsigned long long temp_pair_1[2], temp_pair_2[2];
	double i_64 = pow(2, 64), uniform_variable = 1.0;
	int index_j = 0, m_bits = 0, index_i = 0, index_d_array = 0, temp_y, temp;
	unsigned long long hash_uniform_outcom[2] = {0};
	unsigned long long hash_xor_outcom[2] = {0};
	unsigned long long hash_rank_outcom[2] = {0};
	m_bits = log2(m);
	// insert
	for (j = 0; j < m; ++j)
	{
		permu_origin[j] = j;
	}
	for (i = 0; i < database_rows; ++i)
	{
		uniform_variable = 1.0;
		MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_array_d, hash_uniform_outcom);
		index_d_array = (hash_uniform_outcom[0] >> (64 - m_bits));
		half_xor_sketch[m * w + index_d_array] += 1;
		memcpy(permu, permu_origin, 4 * m);
		for (j = m - 1; j >= 0; --j)
		{
			MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_uniform[j], hash_uniform_outcom);
			uniform_variable = uniform_variable * pow(double(hash_uniform_outcom[0]) / i_64, 1.0 / (j + 1));
			x = poisson(uniform_variable, lambda);
			if (x == 0)
			{
				break;
			}
			temp_y = floor((double(hash_uniform_outcom[1]) / i_64) * (j + 1));
			if (temp_y >= (j + 1)) {temp_y = j;} else if (temp_y < 0) {temp_y = 0;}
			{temp = permu[temp_y]; permu[temp_y] = permu[j]; permu[j] = temp;}
			for (k = 1; k <= x; ++k)
			{
				temp_pair_1[0] = data_values[i];
				temp_pair_1[1] = k;
				MurmurHash3_x64_128(temp_pair_1, 16, half_xor_seed_rank[permu[j]], hash_rank_outcom);
				xor_rank = rank_bits(hash_rank_outcom[0], w);
				temp_pair_2[0] = primary_keys[i];
				temp_pair_2[1] = (k << 48) + permu[j];
				MurmurHash3_x64_128(temp_pair_2, 16, half_xor_seed_xor[permu[j]], hash_xor_outcom);
				half_xor_sketch[permu[j] * w + xor_rank] ^= (hash_xor_outcom[0] >> 63);
			}
		}
	}
	// delete
	for (i = 0; i < (unsigned long long)(database_rows * ratio); ++i)
	{
		uniform_variable = 1.0;
		MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_array_d, hash_uniform_outcom);
		index_d_array = (hash_uniform_outcom[0] >> (64 - m_bits));
		half_xor_sketch[m * w + index_d_array] -= 1;
		memcpy(permu, permu_origin, 4 * m);
		for (j = m - 1; j >= 0; --j)
		{
			MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_uniform[j], hash_uniform_outcom);
			uniform_variable = uniform_variable * pow(double(hash_uniform_outcom[0]) / i_64, 1.0 / (j + 1));
			x = poisson(uniform_variable, lambda);
			if (x == 0)
			{
				break;
			}
			temp_y = floor((double(hash_uniform_outcom[1]) / i_64) * (j + 1));
			if (temp_y >= (j + 1)) {temp_y = j;} else if (temp_y < 0) {temp_y = 0;}
			{temp = permu[temp_y]; permu[temp_y] = permu[j]; permu[j] = temp;}
			for (k = 1; k <= x; ++k)
			{
				temp_pair_1[0] = data_values[i];
				temp_pair_1[1] = k;
				MurmurHash3_x64_128(temp_pair_1, 16, half_xor_seed_rank[permu[j]], hash_rank_outcom);
				xor_rank = rank_bits(hash_rank_outcom[0], w);
				temp_pair_2[0] = primary_keys[i];
				temp_pair_2[1] = (k << 48) + permu[j];
				MurmurHash3_x64_128(temp_pair_2, 16, half_xor_seed_xor[permu[j]], hash_xor_outcom);
				half_xor_sketch[permu[j] * w + xor_rank] ^= (hash_xor_outcom[0] >> 63);
			}
		}
	}
	delete[] permu;
	delete[] permu_origin;
}

void half_xor_generate_sketch_stochastic_averaging(int m, unsigned long long n, int w, double lambda, unsigned long long * data_values, unsigned long long * primary_keys)
{
	half_xor_sketch = new unsigned[m * w + m]();
	unsigned * permu = new unsigned[m]();
	unsigned long long i, j, k, xor_rank = 0, r_base = 10000000000, x;
	unsigned long long temp_pair_1[2], temp_pair_2[2];
	double i_64 = pow(2, 64), uniform_variable = 1.0;
	int index_j = 0, m_bits = 0, index_i = 0, index_d_array = 0, temp_y, temp;
	unsigned long long hash_uniform_outcom[2] = {0};
	unsigned long long hash_xor_outcom[2] = {0};
	unsigned long long hash_rank_outcom[2] = {0};
	m_bits = log2(m);
	for (i = 0; i < database_rows; ++i)
	{
		MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_array_d, hash_uniform_outcom);
		index_d_array = (hash_uniform_outcom[0] >> (64 - m_bits));
		half_xor_sketch[m * w + index_d_array] += 1;
		uniform_variable = (double(hash_uniform_outcom[1]) / i_64);
		x = poisson(uniform_variable, m * lambda);
		if (x == 0) continue;
		for (k = 1; k <= x; ++k)
		{
			temp_pair_1[0] = data_values[i];
			temp_pair_1[1] = k;
			MurmurHash3_x64_128(temp_pair_1, 16, half_xor_seed_uniform[0], hash_rank_outcom);
			xor_rank = rank_bits(hash_rank_outcom[0], w);
			temp_y = (hash_rank_outcom[1] >> (64 - m_bits));
			temp_pair_2[0] = primary_keys[i];
			temp_pair_2[1] = (k << 48) + temp_y;
			MurmurHash3_x64_128(temp_pair_2, 16, half_xor_seed_xor[temp_y], hash_xor_outcom);
			half_xor_sketch[temp_y * w + xor_rank] ^= (hash_xor_outcom[0] >> 63);
		}
	}
	delete[] permu;
}

void half_xor_generate_sketch_bucket_hashing(int m, unsigned long long n, int w, unsigned long long * data_values, unsigned long long * primary_keys)
{
	half_xor_sketch = new unsigned[m * w + m]();
	unsigned * permu = new unsigned[m]();
	unsigned long long i, j, k, xor_rank = 0, r_base = 10000000000, x;
	unsigned long long temp_pair;
	double i_64 = pow(2, 64), uniform_variable = 1.0;
	int index_j = 0, m_bits = 0, index_i = 0, index_d_array = 0, temp_y, temp;
	unsigned long long hash_uniform_outcom[2] = {0};
	unsigned long long hash_xor_outcom[2] = {0};
	unsigned long long hash_rank_outcom[2] = {0};
	m_bits = log2(m);
	for (i = 0; i < database_rows; ++i)
	{
		MurmurHash3_x64_128(&data_values[i], 8, half_xor_seed_array_d, hash_uniform_outcom);
		index_d_array = (hash_uniform_outcom[0] >> (64 - m_bits));
		half_xor_sketch[m * w + index_d_array] += 1;
		x = rank_bits(hash_uniform_outcom[1], w);
		temp_y = (hash_uniform_outcom[1] >> (64 - m_bits));
		temp_pair = primary_keys[i];
		MurmurHash3_x64_128(&temp_pair, 8, half_xor_seed_xor[0], hash_xor_outcom);
		half_xor_sketch[temp_y * w + x] ^= (hash_xor_outcom[0] >> 63);
	}
	delete[] permu;
}

int main()
{
	unsigned long long n = 10000, i;
	int m = 1024, w = 32;
	double estimate = 0, ratio = 0.5, lambda = 1.0 / double(m);
	srand(33434657);
	init_half_xor_seed(1024);
	unsigned long long * data_values = new unsigned long long[10*n];
	unsigned long long * primary_values = new unsigned long long[10*n];
	database_rows = generate_dataset(data_values, primary_values, n);
	cout << endl << setw(40) << "real_NDV:" << setw(20) << n << endl;
	cout << setw(40) << "storage:" << setw(20) << "8KB" << endl;
	cout << setw(40) << "w:" << setw(20) << w << endl;
	cout << setw(40) << "lambda:" << setw(20) << "1 / m" << endl << endl;
	cout << setw(40) << " " << setw(20) << "Half-Xor-EZ" << setw(20) << "Half-Xor-IVW" << endl;

	// half xor (order statistics)
	cout << setw(40) << "Half-Xor (order statistics)";
	half_xor_generate_sketch_order_statistics(m, n, w, lambda, data_values, primary_values);
	estimate = EZ_estimate(1024, 32, 1.0 / 1024.0);
	cout << setw(20) << estimate;
	estimate = IVW_estimate(1024, 32, 1.0 / 1024.0, estimate);
	cout << setw(20) << estimate << endl;
	delete[] half_xor_sketch;

	// half xor delete (order statistics)
	// cout << setw(40) << "Half-Xor delete 50% (order statistics)";
	// half_xor_generate_sketch_order_statistics_insert_and_delete(0.5, m, n, w, lambda, data_values, primary_values);
	// estimate = EZ_estimate(1024, 32, 1.0 / 1024.0);
	// cout << setw(20) << estimate;
	// estimate = IVW_estimate(1024, 32, 1.0 / 1024.0, estimate);
	// cout << setw(20) << estimate << endl;
	// delete[] half_xor_sketch;

	// half xor (stochastic averaging)
	cout << setw(40) << "Half-Xor (stochastic averaging)"; 
	half_xor_generate_sketch_stochastic_averaging(m, n, w, lambda, data_values, primary_values);
	estimate = EZ_estimate(1024, 32, 1.0 / 1024.0);
	cout << setw(20) << estimate;
	estimate = IVW_estimate(1024, 32, 1.0 / 1024.0, estimate);
	cout << setw(20) << estimate << endl;
	delete[] half_xor_sketch;

	// half xor (bucket hashing)
	cout << setw(40) << "Half-Xor (bucket hashing)"; 
	half_xor_generate_sketch_bucket_hashing(m, n, w, data_values, primary_values);
	estimate = EZ_estimate(1024, 32, 1.0 / 1024.0);
	cout << setw(20) << estimate;
	estimate = IVW_estimate(1024, 32, 1.0 / 1024.0, estimate);
	cout << setw(20) << estimate << endl << endl;
	delete[] half_xor_sketch;

	// half xor 
	// cout << setw(40) << "Half-Xor"; 
	// half_xor_generate_sketch(m, n, w, lambda, data_values, primary_values);
	// estimate = EZ_estimate(1024, 32, 1.0 / 1024.0);
	// cout << setw(20) << estimate;
	// estimate = IVW_estimate(1024, 32, 1.0 / 1024.0, estimate);
	// cout << setw(20) << estimate << endl << endl;
	// delete[] half_xor_sketch;

	delete[] half_xor_seed_uniform;
	delete[] half_xor_seed_xor;
	delete[] half_xor_seed_rank;
	delete[] data_values;
	delete[] primary_values;
	return 0;
}