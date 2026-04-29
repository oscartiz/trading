import asyncio
import json
import time
import polars as pl
import websockets

async def collect_ws_data(duration_seconds=120, output_file="real_snapshots.csv", num_levels=10):
    print(f"Collecting live data from Binance for {duration_seconds} seconds...")
    url = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
    start_time = time.time()
    snapshots = []
    
    async with websockets.connect(url) as ws:
        while time.time() - start_time < duration_seconds:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                
                # The payload has 'bids' and 'asks'
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                # Format: timestamp, ask_p0, ask_v0, bid_p0, bid_v0, ...
                # Use the server time if available, or local time
                ts = int(time.time() * 1000)
                row = [ts]
                for i in range(num_levels):
                    if i < len(asks) and i < len(bids):
                        row.extend([float(asks[i][0]), float(asks[i][1]), float(bids[i][0]), float(bids[i][1])])
                    else:
                        row.extend([0.0, 0.0, 0.0, 0.0])
                
                snapshots.append(row)
                
                if len(snapshots) % 100 == 0:
                    print(f"Collected {len(snapshots)} snapshots... ({(time.time() - start_time):.1f}s)")
            except Exception as e:
                print(f"Error: {e}")
                break
                
    print(f"Finished collecting {len(snapshots)} snapshots. Saving to {output_file}...")
    
    columns = ["timestamp"]
    for i in range(num_levels):
        columns.extend([f"ask_price_{i}", f"ask_vol_{i}", f"bid_price_{i}", f"bid_vol_{i}"])
        
    df = pl.DataFrame(snapshots, schema=columns, orient="row")
    df.write_csv(output_file)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(collect_ws_data(duration_seconds=120, output_file="real_snapshots.csv"))
