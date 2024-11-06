#include <stdio.h>     // fopen, fread, fclose, printf, fseek, ftell
#include <math.h>      // log, exp
#include <stdlib.h>    // free, realloc
#include <time.h>      // struct timespec, clock_gettime, CLOCK_REALTIME
#include <errno.h>


double geomean_even_split(unsigned char *s, size_t n) {
    double answer = 0.0;
    #pragma omp parallel for reduction(+:answer)
    for (size_t i = 0; i < n; i++) {
        if (s[i] > 0) {
            answer += log(s[i]) / n;
        }
    }
    return exp(answer);
}

double geomean_task_queue(unsigned char *s, size_t n) {
    double answer = 0.0;
    int index = 0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        while (1) {
            int i;
            #pragma omp atomic capture
            i = index++;
            if (i >= n) break;

            if (s[i] > 0) {
                local_sum += log(s[i]) / n;
            }
        }
        #pragma omp atomic
        answer += local_sum;
    }
    return exp(answer);
}

double geomean_guided(unsigned char *s, size_t n) {
    double answer = 0.0;
    #pragma omp parallel for schedule(guided) reduction(+:answer)
    for (size_t i = 0; i < n; i++) {
        if (s[i] > 0) {
            answer += log(s[i]) / n;
        }
    }
    return exp(answer);
}

double geomean_task_queue_large(unsigned char *s, size_t n, int K) {
    double answer = 0.0;
    int index = 0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        while (1) {
            int start;
            #pragma omp atomic capture
            start = index += K;

            if (start - K >= n) break;

            for (int i = start - K; i < start && i < n; i++) {
                if (s[i] > 0) {
                    local_sum += log(s[i]) / n;
                }
            }
        }
        #pragma omp atomic
        answer += local_sum;
    }
    return exp(answer);
}

double geomean_atomic_non_parallel(unsigned char *s, size_t n) {
    double answer = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (s[i] > 0) {
            #pragma omp atomic
            answer += log(s[i]) / n;
        }
    }
    return exp(answer);
}

double geomean_many_to_few_atomic(unsigned char *s, size_t n) {
    double answer = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for nowait
        for (size_t i = 0; i < n; i++) {
            if (s[i] > 0) {
                local_sum += log(s[i]) / n;
            }
        }
        #pragma omp atomic
        answer += local_sum;
    }
    return exp(answer);
}

double geomean(unsigned char *s, size_t n) {
    return geomean_even_split(s, n); 
    //return geomean_task_queue(s, n); 
    //return geomean_task_queue_large(s, n, 100);
    //return geomean_guided(s, n);
    //return geomean_atomic_non_parallel(s, n);
    //return geomean_many_to_few_atomic(s, n);
}

/// nanoseconds that have elapsed since 1970-01-01 00:00:00 UTC
long long nsecs() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec*1000000000 + t.tv_nsec;
}

/// reads arguments and invokes geomean; should not require editing
int main(int argc, char *argv[]) {
    // step 1: get the input array (the bytes in this file)
    char *s = NULL;
    size_t n = 0;
    for(int i=1; i<argc; i+=1) {
        // add argument i's file contents (or string value) to s
        FILE *f = fopen(argv[i], "rb");
        if (f) { // was a file; read it
            fseek(f, 0, SEEK_END); // go to end of file
            size_t size = ftell(f); // find out how many bytes in that was
            fseek(f, 0, SEEK_SET); // go back to beginning
            s = realloc(s, n+size); // make room
            fread(s+n, 1, size, f); // append this file on end of others
            fclose(f);
            n += size; // not new size
        } else { // not a file; treat as a string
            errno = 0; // clear the read error
        }
    }

    // step 2: invoke and time the geometric mean function
    long long t0 = nsecs();
    double answer = geomean((unsigned char*) s,n);
    long long t1 = nsecs();

    free(s);

    // step 3: report result
    printf("%lld ns to process %zd characters: %g\n", t1-t0, n, answer);
}

