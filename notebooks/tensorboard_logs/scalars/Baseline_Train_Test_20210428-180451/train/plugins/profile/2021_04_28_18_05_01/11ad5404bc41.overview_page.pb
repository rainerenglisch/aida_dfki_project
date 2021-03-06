?	;??u??[@;??u??[@!;??u??[@	¡??ܛ??¡??ܛ??!¡??ܛ??"w
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
XŁ?2F@yξ???K@?""
ArgMaxArgMax?e?????!?e?????".
IteratorGetNext/_16_Recvk? gɽ?!z???b`??"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop<??
???!{?i\????0"C
%Baseline_Train_Test/enc_output/MatMulMatMul?(O???!??< ??0"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam?r~)???!5?D??B??"a
Egradient_tape/Baseline_Train_Test/dec_dense/Tensordot/MatMul/MatMul_1MatMula????o??!???	??"L
.Baseline_Train_Test/dec_dense/Tensordot/MatMulMatMul0??l??!ܙ??????0"a
Cgradient_tape/Baseline_Train_Test/dec_dense/Tensordot/MatMul/MatMulMatMul]C{ݱ???!N?????0"&
CudnnRNNCudnnRNN???U??!ԻP#I??"?
qloss_func/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?1??b$??!c??dF"??Q      Y@Y??????/@a? ? U@qi?pz?t??y??Z4?x?"?

both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?39.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 