#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#define DATASET_SIZE 200

typedef struct
{
    float spotPrice;      // Current market price of the stock
    float strikePrice;    // Strike price of the option
    float riskFreeRate;   // Risk-free interest rate
    float volatility;     // Volatility of the stock
    float timeToMaturity; // Time to expiration of the option
    int optionType;       // 0 for call, 1 for put
} OptionData;

OptionData generateRandomData()
{
    OptionData data;
    data.spotPrice = (rand() % 100) + 50;
    data.strikePrice = data.spotPrice + (rand() % 20 - 10);
    data.riskFreeRate = (float)rand() / (float)(RAND_MAX) * 0.05;
    data.volatility = (float)rand() / (float)(RAND_MAX) * 0.3;
    data.timeToMaturity = (float)rand() / (float)(RAND_MAX) * 2;
    data.optionType = rand() % 2;
    return data;
}

#define inv_sqrt_2xPI 0.39894228040143270286
float CNDF(float InputX)
{
    int sign;
    float OutputX;
    float xInput;
    float xNPrimeofX;
    float expValues;
    float xK2;
    float xK2_2, xK2_3;
    float xK2_4, xK2_5;
    float xLocal, xLocal_1;
    float xLocal_2, xLocal_3;
    // Check for negative value of InputX
    if (InputX < 0.0)
    {
        InputX = -InputX;
        sign = 1;
    }
    else
        sign = 0;
    xInput = InputX;
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;
    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;
    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal = xLocal_1 * xNPrimeofX;
    xLocal = 1.0 - xLocal;
    OutputX = xLocal;
    if (sign)
    {
        OutputX = 1.0 - OutputX;
    }
    return OutputX;
}

float blackScholes(float sptprice, float strike, float rate, float volatility,
                   float otime, int otype)
{
    float OptionPrice;
    // local private working variables for the calculation
    float xStockPrice;
    float xStrikePrice;
    float xRiskFreeRate;
    float xVolatility;
    float xTime;
    float xSqrtTime;
    float logValues;
    float xLogTerm;
    float xD1;
    float xD2;
    float xPowerTerm;
    float xDen;
    float d1;
    float d2;
    float FutureValueX;
    float NofXd1;
    float NofXd2;
    float NegNofXd1;
    float NegNofXd2;
    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;
    xTime = otime;
    xSqrtTime = sqrt(xTime);
    logValues = log(sptprice / strike);
    xLogTerm = logValues;
    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;
    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;
    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 - xDen;
    d1 = xD1;
    d2 = xD2;
    NofXd1 = CNDF(d1);
    NofXd2 = CNDF(d2);
    FutureValueX = strike * (exp(-(rate) * (otime)));
    if (otype == 0)
    {
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    }
    else
    {
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }
    return OptionPrice;
}

typedef struct
{
    OptionData *dataset;
    int start_index;
    int end_index;
} ThreadArg;

void *blackScholesThread(void *arg)
{
    ThreadArg *threadArg = (ThreadArg *)arg;
    OptionData *dataset = threadArg->dataset;
    int start = threadArg->start_index;
    int end = threadArg->end_index;

    for (int i = start; i < end; i++)
    {
        float price = blackScholes(dataset[i].spotPrice, dataset[i].strikePrice,
                                   dataset[i].riskFreeRate, dataset[i].volatility,
                                   dataset[i].timeToMaturity, dataset[i].optionType);
        printf("Thread: %ld, Data point: %d, Option Price: %f\n", (unsigned long)pthread_self(), i, price);
    }
    free(arg);
    return NULL;
}

// void* blackScholesWorker(void *args) {
//     args_t* p_args = (args_t*)args;

//     int start = p_args->start;
//     int end = p_args->end;
//     float *prices = p_args->prices;
//     OptionData *dataset = p_args->dataset;

//     for (int i = start; i < end; i++) {
//         prices[i] = blackScholes(dataset[i].spotPrice, dataset[i].strikePrice,
//                                  dataset[i].riskFreeRate, dataset[i].volatility,
//                                  dataset[i].timeToMaturity, dataset[i].optionType);
//     }

//     return NULL;
// }

// void* impl_parallel(void *args)
// {
//     /* Get the argument struct */
//     args_t* p_args = (args_t*)args;

//     /* Get all the arguments */
//     register int* dest = (int*)(p_args->output);
//     register const int* src0 = (const int*)(p_args->input0);
//     register const int* src1 = (const int*)(p_args->input1);
//     register size_t size = p_args->size / 4;

//     register size_t nthreads = p_args->nthreads - 1;
//     register size_t cpu = p_args->cpu;

//     /* Create all threads */
//     pthread_t tid[nthreads];
//     args_t targs[nthreads];
//     cpu_set_t cpuset[nthreads];

//     /* Assign current CPU to us */
//     tid[0] = pthread_self();

//     /* Affinity */
//     CPU_ZERO(&(cpuset[0]));
//     CPU_SET(cpu, &(cpuset[0]));

//     /* Set affinity */
//     int res_affinity_0 = pthread_setaffinity_np(tid[0], sizeof(cpuset[0]), &(cpuset[0]));

//     /* Amount of work per thread */
//     size_t size_per_thread = size / nthreads;

//     for (int i = 1; i < nthreads; i++) {
//         /* Initialize the argument structure */
//         targs[i].size = size_per_thread;
//         targs[i].input0 = (byte*)(src0 + (i * size_per_thread));
//         targs[i].input1 = (byte*)(src1 + (i * size_per_thread));
//         targs[i].output = (byte*)(dest + (i * size_per_thread));

//         targs[i].cpu = (cpu + i) % nthreads;
//         targs[i].nthreads = nthreads;

//         /* Affinity */
//         CPU_ZERO(&(cpuset[i]));
//         CPU_SET(targs[i].cpu, &(cpuset[i]));

//         /* Set affinity */
//         int res = pthread_create(&tid[i], NULL, worker, (void*)&targs[i]);
//         int res_affinity = pthread_setaffinity_np(tid[i], sizeof(cpuset[i]), &(cpuset[i]));
//     }

//     /* Perform one portion of the work */
//     for (int i = 0; i < size_per_thread; i++) {
//         dest[i] = src0[i] + src1[i];
//     }

//     /* Perform trailing elements */
//     int remaining = size % nthreads;
//     for (int i = size - remaining; i < size; i++) {
//         dest[i] = src0[i] + src1[i];
//     }

//     /* Wait for all threads to finish execution */
//     for (int i = 0; i < nthreads; i++) {
//         pthread_join(tid[i], NULL);
//     }

//     /* Done */
//     return 0;
// }



// void* blackScholesThread(void* arg) {
//     // OptionData* dataset = (OptionData*)arg;
//         int index = *((int*)arg);
//     int chunk_size = DATASET_SIZE / num_threads;
//     int start = index * chunk_size;
//     int end = (index == num_threads - 1) ? DATASET_SIZE : start + chunk_size;

//     for (int i = start; i < end; i++) { // Assuming 2 threads for simplicity
//         float price = blackScholes(dataset[i].spotPrice, dataset[i].strikePrice,
//                                    dataset[i].riskFreeRate, dataset[i].volatility,
//                                    dataset[i].timeToMaturity, dataset[i].optionType);
//         printf("Thread: %ld, Data point: %d, Option Price: %f\n", (unsigned long)pthread_self(), i, price);
//     }
//     free(arg);
//     return NULL;
// }

// int main() {
//     srand(time(NULL)); // Seed for random number generation
//     OptionData dataset[DATASET_SIZE];

//     // Generate dataset
//     for (int i = 0; i < DATASET_SIZE; i++) {
//         dataset[i] = generateRandomData();
//     }

//     // Iterate over dataset and calculate option prices
//     for (int i = 0; i < DATASET_SIZE; i++) {
//         float price = blackScholes(dataset[i].spotPrice, dataset[i].strikePrice,
//                                    dataset[i].riskFreeRate, dataset[i].volatility,
//                                    dataset[i].timeToMaturity, dataset[i].optionType);
//         printf("Option Price for data point %d: %f\n", i, price);
//     }

//     pthread_t threads[2]; // Creating two threads

//     // Launch threads
//     for (int i = 0; i < 2; i++) {
//         pthread_create(&threads[i], NULL, blackScholesThread, (void*)(dataset + i * (DATASET_SIZE / 2)));
//     }

//     // Wait for threads to complete
//     for (int i = 0; i < 2; i++) {
//         pthread_join(threads[i], NULL);
//     }

//     return 0;

// }

int get_num_threads()
{
    long num_cores;

    num_cores = sysconf(_SC_NPROCESSORS_ONLN); // Get the number of online processors

    if (num_cores == -1)
    {
        // In case of an error, default to a conservative number of threads
        return 2;
    }

    return (int)num_cores;
}
int main()
{
    srand(time(NULL)); // Seed for random number generation
    OptionData dataset[DATASET_SIZE];

    // Generate dataset
    for (int i = 0; i < DATASET_SIZE; i++)
    {
        dataset[i] = generateRandomData();
    }

    // Start timing for SISD
    clock_t start_sisd = clock();
    // Perform single-threaded Black-Scholes computation
    for (int i = 0; i < DATASET_SIZE; i++)
    {
        float price = blackScholes(dataset[i].spotPrice, dataset[i].strikePrice,
                                   dataset[i].riskFreeRate, dataset[i].volatility,
                                   dataset[i].timeToMaturity, dataset[i].optionType);
        // Optionally print the results or store them for later use
    }
    // Stop timing for SISD
    clock_t end_sisd = clock();
    double time_sisd = (double)(end_sisd - start_sisd) / CLOCKS_PER_SEC;

    // Start timing for MIMD
    clock_t start_mimd = clock();
    // pthread_t threads[2]; // Creating two threads
    // Launch threads
    // for (int i = 0; i < 2; i++) {
    //     pthread_create(&threads[i], NULL, blackScholesThread, (void*)(dataset + i * (DATASET_SIZE / 2)));
    // }
    int num_threads = get_num_threads();
    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++)
    {
        ThreadArg *args = malloc(sizeof(ThreadArg));
        args->dataset = dataset;
        args->start_index = i * (DATASET_SIZE / num_threads);
        args->end_index = (i == num_threads - 1) ? DATASET_SIZE : (i + 1) * (DATASET_SIZE / num_threads);
        pthread_create(&threads[i], NULL, blackScholesThread, args);
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // Wait for threads to complete
    for (int i = 0; i < 2; i++)
    {
        pthread_join(threads[i], NULL);
    }
    // impl_parallel(&args);
    // Stop timing for MIMD
    clock_t end_mimd = clock();
    double time_mimd = (double)(end_mimd - start_mimd) / CLOCKS_PER_SEC;

    // Print the results
    printf("Time taken for SISD: %f seconds\n", time_sisd);
    printf("Time taken for MIMD: %f seconds\n", time_mimd);

    return 0;
}