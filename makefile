hll-fd: MurmurHash3.o HLL-FD.o
	g++ -o hll-fd HLL-FD.o MurmurHash3.o -pthread
hll-prob: MurmurHash3.o HLL-Prob.o
	g++ -o hll-prob HLL-Prob.o MurmurHash3.o -pthread
half-xor: MurmurHash3.o Half-Xor.o
	g++ -o half-xor Half-Xor.o MurmurHash3.o -pthread
