from multiprocessing import Process, Queue
import time
import random

# ---------- Stage 1: Price Difference ----------
def stage_diff(input_q, output_q):
    prev = None
    index = 0
    while True:
        item = input_q.get()
        if item == "DONE":
            output_q.put("DONE")
            break
        if prev is not None:
            diff = item - prev
            output_q.put((index, diff))
            index += 1
        prev = item

# ---------- Stage 2: Gain and Loss ----------
def stage_gain_loss(input_q, output_q):
    while True:
        item = input_q.get()
        if item == "DONE":
            output_q.put("DONE")
            break
        i, diff = item
        gain = max(diff, 0)
        loss = max(-diff, 0)
        output_q.put((i, gain, loss))

# ---------- Stage 3: EMA of Gain and Loss ----------
def stage_avg(input_q, output_q, period):
    gain_hist = []
    loss_hist = []
    EMA_gain = None
    EMA_loss = None
    while True:
        item = input_q.get()
        if item == "DONE":
            output_q.put("DONE")
            break
        i, gain, loss = item
        if i < period:
            gain_hist.append(gain)
            loss_hist.append(loss)
            continue
        elif i == period:
            EMA_gain = sum(gain_hist + [gain]) / period
            EMA_loss = sum(loss_hist + [loss]) / period
        else:
            EMA_gain = (EMA_gain * (period - 1) + gain) / period
            EMA_loss = (EMA_loss * (period - 1) + loss) / period

        output_q.put((i, EMA_gain, EMA_loss))

# ---------- Stage 4: RSI Calculation ----------
def stage_rsi(input_q, output_q):
    while True:
        item = input_q.get()
        if item == "DONE":
            output_q.put("DONE")
            break
        i, ag, al = item
        rs = ag / al if al != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))
        output_q.put((i, rsi))

# ---------- Pipeline Driver ----------
def rsi_pipeline(close_prices, period=14):
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    q_out = Queue()

    # Spawn stage processes
    p1 = Process(target=stage_diff, args=(q1, q2))
    p2 = Process(target=stage_gain_loss, args=(q2, q3))
    p3 = Process(target=stage_avg, args=(q3, q4, period))
    p4 = Process(target=stage_rsi, args=(q4, q_out))

    for p in [p1, p2, p3, p4]:
        p.start()

    # Feed input prices
    for cp in close_prices:
        q1.put(cp)
    q1.put("DONE")

    # Collect output RSI values
    result = [None] * len(close_prices)
    while True:
        item = q_out.get()
        if item == "DONE":
            break
        i, rsi = item
        result[i] = rsi

    for p in [p1, p2, p3, p4]:
        p.join()

    return result

# ---------- Main ----------
if __name__ == "__main__":
    # Large price dataset for stress test
    large_prices = [100 + random.uniform(-1, 1) for _ in range(1000000)]

    start_time = time.time()
    rsi_result = rsi_pipeline(large_prices, period=14)
    end_time = time.time()

    print(f" Parallel RSI (Pipelined) Execution Time: {end_time - start_time:.4f} seconds")
    # Optional: print first 10 RSI values (excluding None values)
    print("Sample RSI values:", [r for r in rsi_result if r is not None][:10])
