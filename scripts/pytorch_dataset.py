import torch
import pyarrow.parquet as pq

class ForexIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, parquet_file, seq_len=128, batch_size=64):
        super().__init__()
        self.parquet_file = parquet_file
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        table = pq.ParquetFile(self.parquet_file)
        for batch in table.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            values = df[["open","high","low","close","volume"]].values
            for i in range(len(values)-self.seq_len):
                x = values[i:i+self.seq_len]
                y = values[i+self.seq_len]
                yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
