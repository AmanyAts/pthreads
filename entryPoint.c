#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>


#define DATASET_SIZE 201

typedef struct
{
    float* dest;            // Array to store option prices
    float* spotPrice;       // Pointer to spotPrice array
    float* strikePrice;     // Pointer to strikePrice array
    float* riskFreeRate;    // Pointer to riskFreeRate array
    float* volatility;      // Pointer to volatility array
    float* timeToMaturity; // Pointer to timeToMaturity array
    int* optionType;        // Pointer to optionType array
    size_t size;            // Number of data points to process
    int num_threads;        // Number of threads
    int cpu;
} ThreadArgs;

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


void* blackScholesWorker(void* args) {
    // Parse the arguments structure
    ThreadArgs* p_args = (ThreadArgs*)args;

    // Get all the arguments
    register       float* dest = (      float*)(p_args->dest);
    register const float*   spotPrice = (const float*)(p_args->spotPrice);
    register const float*   strikePrice = (const float*)(p_args->strikePrice);
    register const float*   riskFreeRate = (const float*)(p_args->riskFreeRate);
    register const float*   volatility = (const float*)(p_args->volatility);
    register const float*   timeToMaturity = (const float*)(p_args->timeToMaturity);
    register const int*   optionType = (const int*)(p_args->optionType);
    register size_t size = p_args->size;

    for (int i = 0; i < size; i++) {
        dest[i] = blackScholes(spotPrice[i], strikePrice[i], riskFreeRate[i],
                               volatility[i], timeToMaturity[i], optionType[i]);
    }

    return NULL;
}

void impl_parallel(float* output,  float* input0,  float* input1,  float* input2,
                    float* input3,  float* input4,  int* input5, size_t size, int num_threads) {
    // Create an array of pthread_t to hold thread IDs
    pthread_t tid[num_threads];

    // Create an array of args_t to hold thread arguments
    ThreadArgs thread_args[num_threads];

    // Calculate the size of work for each thread
    size_t chunk_size = size / num_threads;
    size_t remaining = size % num_threads;
    // Initialize CPU cores to be used by threads
    int cpu_cores[num_threads];
    for (int i = 0; i < num_threads; i++) {
         printf("The value of  is: %d\n", num_threads);

        cpu_cores[i] = i ; // Assign cores sequentially
        printf("The value of num is: %d\n", cpu_cores[i]);

    }

    // Create and run threads
    for (int i = 0; i < num_threads; i++) {
        
         printf("i: %d\n", i);
        // Set thread arguments
        thread_args[i].size = (i < remaining) ? chunk_size + 1 : chunk_size;
        thread_args[i].dest = output + i * chunk_size;
        thread_args[i].spotPrice = input0 + i * chunk_size;
        thread_args[i].strikePrice = input1 + i * chunk_size;
        thread_args[i].riskFreeRate = input2 + i * chunk_size;
        thread_args[i].volatility = input3 + i * chunk_size;
        thread_args[i].timeToMaturity = input4 + i * chunk_size;
        thread_args[i].optionType = input5 + i * chunk_size;
        thread_args[i].num_threads = num_threads;
        thread_args[i].cpu = cpu_cores[i];

        // Create and run the thread
        pthread_create(&tid[i], NULL, blackScholesWorker, &thread_args[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
}

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

     // Initialize arrays for input and output
    float spotPrice[DATASET_SIZE];
    float strikePrice[DATASET_SIZE];
    float riskFreeRate[DATASET_SIZE];
    float volatility[DATASET_SIZE];
    float timeToMaturity[DATASET_SIZE];
    int optionType[DATASET_SIZE];
    float optionPrice[DATASET_SIZE];

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

   // Set the number of threads you want to use
    int num_threads = get_num_threads(); // You can adjust this as needed

    // Perform Black-Scholes calculations in parallel
    impl_parallel(optionPrice, spotPrice, strikePrice, riskFreeRate, volatility,
                  timeToMaturity, optionType, DATASET_SIZE, num_threads);


    // Stop timing for MIMD
    clock_t end_mimd = clock();
    double time_mimd = (double)(end_mimd - start_mimd) / CLOCKS_PER_SEC;

    // Print the results
    printf("Time taken for SISD: %f seconds\n", time_sisd);
    printf("Time taken for MIMD: %f seconds\n", time_mimd);

    return 0;
}