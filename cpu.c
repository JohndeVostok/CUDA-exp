#include <stdio.h>
#include <sys/time.h>

static long long ustime(void) {
	struct timeval tv;
	long long ust;
	gettimeofday(&tv, NULL);
	ust = ((long)tv.tv_sec)*1000000;
	ust += tv.tv_usec;
	return ust;
}

double getPi(int n) {
	double s = 0;
	for (int i = 0; i < n; i++) {
		double t = (2 * i + 1) / (n * 2.0);
		s += 4 / (1 + t * t);
	}
	return s / n;
}

int main() {
	long long op, ed;
	op = ustime();
	double ans = getPi(200000000);
	ed = ustime();
	printf("Pi: %f\nTime: %f\n", ans, (double)((ed - op) / 1000000.0));
}
