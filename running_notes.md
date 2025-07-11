## Late night or early morning notes about dataloading for LLMs

- Reading thru the lingua codebase. Even this uses the producer-consumer model for designing efficient dataloaders such that the GPUs remain busy while multiple CPU workers are prefetching data from the disk.
- So I went down the rabbit hole of tokenizers again. This time came out with some actionable information.
	+ So it is crucial to get the tokenizer.model file from huggingface or any other source. 
	+ We then use the `SentencePieceProcessor` from the `SentencePiece` library to load the vocab and other data from the model file.
	+ lingua does the same as described above.
- So after the rabbit hole, if you want to minrepro (minimal reproducible), then just create a `build_tokenizer` func that returns a `SentencePieceTokenizer` using the steps described above.
- The dataloading logic uses a lot of `State` to determine where each of the underlying steps are. It also allows use to resume the work incase of any failure as the `State`(s) will be written and saved.
- The lingua codebase uses a `contextmanager` from contextlib to manage the iteration. It needs the following things:
	+ `Queue`: To put the work units in. (Producer does this) and to process the work units (Consumer).
	+ `Stop_event`: To announce when the iteration is done and no more work can be produced.
