// rsi_serial.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PERIOD 14
#define N 1000000  // 1 million prices

float* generate_random_prices(int n) {
    float* prices = malloc(sizeof(float) * n);
    if (!prices) return NULL;

    prices[0] = 100.0f;
    for (int i = 1; i < n; ++i) {
        float change = ((rand() % 2001) - 1000) / 1000.0f; // [-1.0, 1.0]
        prices[i] = prices[i - 1] + change;
    }
    return prices;
}

void calc_rsi_serial(float* prices, float* rsi, int n, int period) {
    float gain = 0.0f, loss = 0.0f;

    for (int i = 1; i <= period; ++i) {
        float change = prices[i] - prices[i - 1];
        if (change > 0) gain += change;
        else loss -= change;
    }

    float avg_gain = gain / period;
    float avg_loss = loss / period;

    for (int i = period + 1; i < n; ++i) {
        float change = prices[i] - prices[i - 1];
        float g = change > 0 ? change : 0;
        float l = change < 0 ? -change : 0;

        avg_gain = (avg_gain * (period - 1) + g) / period;
        avg_loss = (avg_loss * (period - 1) + l) / period;

        float rs = avg_loss == 0 ? 100.0f : avg_gain / avg_loss;
        rsi[i] = 100.0f - (100.0f / (1.0f + rs));
    }
}

int main() {
    srand(time(NULL));
    float* prices = generate_random_prices(N);
    float* rsi = calloc(N, sizeof(float));

    clock_t start = clock();
    calc_rsi_serial(prices, rsi, N, PERIOD);
    clock_t end = clock();

    printf("Serial Time: %.6f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("Sample RSI: %.2f %.2f %.2f\n", rsi[N/2], rsi[N/2 + 1], rsi[N/2 + 2]);

    free(prices);
    free(rsi);
    return 0;
}
