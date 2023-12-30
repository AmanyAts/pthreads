#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>



// #define DATASET_SIZE 100000

typedef struct {
    float* dest;
    float* spotPrice;
    float* strikePrice;
    float* riskFreeRate;
    float* volatility;
    float* timeToMaturity;
    int* optionType;
    size_t size;
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



int countRecordsInFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    int count = 0;
    char buffer[256]; // Assuming each line will not exceed 255 characters

    while (fgets(buffer, sizeof(buffer), file)) {
        // Check if the line matches the expected format
        if (sscanf(buffer, " { %*f , %*f , %*f , %*f , %*f , %*f , \"%*[CP]\" , %*f , %*f } ,") == 0) {
            count++;
        }
    }

    fclose(file);
    return count;
}


int readDataFromFile(const char* filename, OptionData* dataset, int maxSize) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    int i = 0;
    char optionTypeStr[10]; // Buffer to store the option type string
    while (i < maxSize) {
        if (fscanf(file, " { %f , %f , %f , %*f , %f , %f , \"%[CP]\" , %*f , %*f } ,",
                        &dataset[i].spotPrice,
                        &dataset[i].strikePrice,
                        &dataset[i].riskFreeRate,
                        &dataset[i].volatility,
                        &dataset[i].timeToMaturity,
                        optionTypeStr) == 6) {
            dataset[i].optionType = (optionTypeStr[0] == 'C') ? 0 : 1;
            i++;
        } else {
            break; // Break the loop if the line doesn't match the expected format
        }
    }

    fclose(file);
    return i; // Return the number of records read
}


OptionData* allocateDataset(int size) {
    return (OptionData*)malloc(size * sizeof(OptionData));
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

    for (int i = 0; i < p_args->size; i++) {
        dest[i] = blackScholes(spotPrice[i], strikePrice[i], riskFreeRate[i],
                               volatility[i], timeToMaturity[i], optionType[i]);
    }

    return NULL;
}

int optimal_thread_count_for_workload(size_t dataset_size, int max_threads) {
    int optimal_threads = (int)ceil(dataset_size / 50.0); // Example heuristic
    if (optimal_threads > max_threads) {
        optimal_threads = max_threads;
    }
    printf("ffff %d\n",optimal_threads);
    return optimal_threads > 0 ? optimal_threads : 1;
}

void impl_parallel(ThreadArgs* args) {
    int max_threads = get_num_threads();
    int num_threads = optimal_thread_count_for_workload(args->size, max_threads);

    pthread_t* tid = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* thread_args = malloc(num_threads * sizeof(ThreadArgs));

    if (!tid || !thread_args) {
        perror("Failed to allocate memory for threads");
        if (tid) free(tid);
        if (thread_args) free(thread_args);
        exit(EXIT_FAILURE);
    }

    size_t batch_size = args->size / num_threads;
    size_t remaining = args->size % num_threads;
    size_t start = 0;

    for (int i = 0; i < num_threads; i++) {
        size_t thread_batch_size = batch_size + (i < remaining ? 1 : 0);

        thread_args[i].size = thread_batch_size;
        thread_args[i].dest = args->dest + start;
        thread_args[i].spotPrice = args->spotPrice + start;
        thread_args[i].strikePrice = args->strikePrice + start;
        thread_args[i].riskFreeRate = args->riskFreeRate + start;
        thread_args[i].volatility = args->volatility + start;
        thread_args[i].timeToMaturity = args->timeToMaturity + start;
        thread_args[i].optionType = args->optionType + start;

        pthread_create(&tid[i], NULL, blackScholesWorker, &thread_args[i]);
        start += thread_batch_size;
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }

    free(tid);
    free(thread_args);
}


int main() {
    const char* filename = "/Users/amanyats/Desktop/COE-231/project/optionData.txt";

    // Count the number of records in the file
    int datasetSize = countRecordsInFile(filename);
    if (datasetSize <= 0) {
        fprintf(stderr, "Failed to read data from the file\n");
        return 1;
    }

    // Dynamically allocate memory for the dataset
    OptionData* dataset = allocateDataset(datasetSize);
    if (!dataset) {
        fprintf(stderr, "Failed to allocate memory for the dataset\n");
        return 1;
    }

    // Read data from file
    if (readDataFromFile(filename, dataset, datasetSize) != datasetSize) {
        fprintf(stderr, "Error reading data from file\n");
        free(dataset);
        return 1;
    }


    // Start timing for SISD
    clock_t start_sisd = clock();
    // Perform single-threaded Black-Scholes computation
    for (int i = 0; i < datasetSize; i++)
    {
        float price = blackScholes(dataset[i].spotPrice, dataset[i].strikePrice,
                                   dataset[i].riskFreeRate, dataset[i].volatility,
                                   dataset[i].timeToMaturity, dataset[i].optionType);
        // Optionally print the results or store them for later use
    }
    // Stop timing for SISD
    clock_t end_sisd = clock();
    double time_sisd = (double)(end_sisd - start_sisd) / CLOCKS_PER_SEC;


    // Initialize arrays for input and output
    float* spotPrice = (float*)malloc(datasetSize * sizeof(float));
    float* strikePrice = (float*)malloc(datasetSize * sizeof(float));
    float* riskFreeRate = (float*)malloc(datasetSize * sizeof(float));
    float* volatility = (float*)malloc(datasetSize * sizeof(float));
    float* timeToMaturity = (float*)malloc(datasetSize * sizeof(float));
    int* optionType = (int*)malloc(datasetSize * sizeof(int));
    float* optionPrice = (float*)malloc(datasetSize * sizeof(float));

    if (!spotPrice || !strikePrice || !riskFreeRate || !volatility || !timeToMaturity || !optionType || !optionPrice) {
        fprintf(stderr, "Failed to allocate memory for arrays\n");
        free(spotPrice);
        free(strikePrice);
        free(riskFreeRate);
        free(volatility);
        free(timeToMaturity);
        free(optionType);
        free(optionPrice);
        free(dataset);
        return 1;
    }

    // Populate input arrays from the dataset
    for (int i = 0; i < datasetSize; i++) {
        spotPrice[i] = dataset[i].spotPrice;
        strikePrice[i] = dataset[i].strikePrice;
        riskFreeRate[i] = dataset[i].riskFreeRate;
        volatility[i] = dataset[i].volatility;
        timeToMaturity[i] = dataset[i].timeToMaturity;
        optionType[i] = dataset[i].optionType;
    }

    // Perform Black-Scholes calculations in parallel
    ThreadArgs args;
    args.dest = optionPrice;
    args.spotPrice = spotPrice;
    args.strikePrice = strikePrice;
    args.riskFreeRate = riskFreeRate;
    args.volatility = volatility;
    args.timeToMaturity = timeToMaturity;
    args.optionType = optionType;
    args.size = datasetSize;

    clock_t start_mimd = clock();
    impl_parallel(&args);
    clock_t end_mimd = clock();
    double time_mimd = (double)(end_mimd - start_mimd) / CLOCKS_PER_SEC;
    printf("Time taken for SISD: %f seconds\n", time_sisd);

    printf("Time taken for MIMD: %f seconds\n", time_mimd);

    // Free dynamically allocated memory
    free(spotPrice);
    free(strikePrice);
    free(riskFreeRate);
    free(volatility);
    free(timeToMaturity);
    free(optionType);
    free(optionPrice);
    free(dataset);

    return 0;
}