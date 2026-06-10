import src.dataloader as dataloader

ego, sur = dataloader.load_train(max_files=1)
print(f"Scenarios in this shard: {ego}")
