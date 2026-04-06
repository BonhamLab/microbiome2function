- [ ] GraphConvNodeClassifier.fit and GraphConvNodeClassifier.test method implementation
- [ ] FFNN.fit and FFNN.test method implementation
- [ ] ProteinGraphOnDiskDataset implementation based on FeatureStore and GraphStore (*)
- [ ] ProteinDataset implementation (*)
    (*): note, topology is built in the same way for ProteinGraphOnDiskDataset as for ProteinGraphInMemoryDataset; the only difference is node and edge features that are constructed and written immediately for ProteinGraphInMemoryDataset but batch-wise for ProteinGraphOnDiskDataset. So, I think we need a helper function shared across the two classes for topology construction, and we need some helper for feature preprocessing orhestration that can be made batched through its parameters, which should be shared across all three ProteinGraphInMemoryDataset, ProteinGraphOnDiskDataset, and ProteinDataset
- [ ] Go over the whole codebase and ensure it is stylistically sound
- [ ] Do static analysis of the codebase
- [ ] Ensure logging and docstring presence across the whole codebase
- [ ] Update and freeze the API
- [ ] Do the final README and docs update