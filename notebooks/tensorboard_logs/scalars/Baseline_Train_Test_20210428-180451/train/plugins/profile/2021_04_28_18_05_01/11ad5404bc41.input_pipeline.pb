	;??u??[@;??u??[@!;??u??[@	¡??ܛ??¡??ܛ??!¡??ܛ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6;??u??[@?????J@1??Z?N@A?2??(??I??EC??E@Y?5Φ#???*	^?ICP@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??n????!??$?'L@)??n????1??$?'L@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??}"??!M??_W@)?7??Ø?1<??$?B@:Preprocessing2F
Iterator::Model
???린?!      Y@)?0??Zq?1<.??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?39.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9¡??ܛ??I
XŁ?2F@Qξ???K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????J@?????J@!?????J@      ??!       "	??Z?N@??Z?N@!??Z?N@*      ??!       2	?2??(???2??(??!?2??(??:	??EC??E@??EC??E@!??EC??E@B      ??!       J	?5Φ#????5Φ#???!?5Φ#???R      ??!       Z	?5Φ#????5Φ#???!?5Φ#???b      ??!       JGPUY¡??ܛ??b q
XŁ?2F@yξ???K@