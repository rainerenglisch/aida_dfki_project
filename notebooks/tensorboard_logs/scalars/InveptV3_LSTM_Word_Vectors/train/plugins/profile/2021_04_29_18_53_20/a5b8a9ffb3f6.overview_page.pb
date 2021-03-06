?	?{??m?d@?{??m?d@!?{??m?d@	hI??????hI??????!hI??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?{??m?d@?'??Z @1??fc@A#?=???I?E?~U??Y?~NA~6??*	o??ʱP@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\[%X??!?d??*P@)\[%X??1?d??*P@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?5??
??!????uW@)(ђ?????1?s?g;,=@:Preprocessing2F
Iterator::Model0??\??!      Y@){?V???p?1??v??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9hI??????I ?????@Q?Z%?oW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'??Z @?'??Z @!?'??Z @      ??!       "	??fc@??fc@!??fc@*      ??!       2	#?=???#?=???!#?=???:	?E?~U???E?~U??!?E?~U??B      ??!       J	?~NA~6???~NA~6??!?~NA~6??R      ??!       Z	?~NA~6???~NA~6??!?~NA~6??b      ??!       JGPUYhI??????b q ?????@y?Z%?oW@?"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop8|?%?5??!8|?%?5??0""
ArgMaxArgMax@؜8b???!,?? ?@??"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdams6Z??f??!ɁcZ??"&
CudnnRNNCudnnRNN?`???Z??! ?ٟ?X??"S
5gradient_tape/model/dec_dense/Tensordot/MatMul/MatMulMatMul??1?U???!???PM???0">
 model/dec_dense/Tensordot/MatMulMatMul??z^7??!??9???0"S
7gradient_tape/model/dec_dense/Tensordot/MatMul/MatMul_1MatMul&!?(??!,??
:???"-
IteratorGetNext/_1_Send-?*K???!3?n???"5
model/enc_output/MatMulMatMul??/??!ѭ?????0"C
%gradient_tape/model/enc_output/MatMulMatMul ?F???!i??Q???0Q      Y@Y8??˕@a?D??_W@qw??Ko@y??)?x?a?"?	
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 