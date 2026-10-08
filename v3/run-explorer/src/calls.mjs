import {readRun} from './dashboard.mjs';
export function buildCalls(archive) {
  const run=readRun(archive),calls=[];
  let phase='';
  const jobs=new Set();
  for(const event of [...run.events].sort((a,b)=>a.sequence-b.sequence)) {
    let source,target,operation;
    if(event.event==='stage_started'){phase=event.stage;source='orchestrator';target=phase;operation=event.attempt>1?'Run revised task':'Run task';}
    else if(event.event==='llm_decision_started'&&event.stage?.startsWith('review.')){source='orchestrator';target='reviewer';operation='Review task';}
    else if(event.event==='local_job_process_started'){
      if(jobs.has(event.job_id))continue;jobs.add(event.job_id);
      source=phase;target='worker';operation={verify:'Verify candidate',train:'Train candidate',infer:'Run predictions'}[event.operation]??event.operation;
    }
    if(source&&target)calls.push({id:`call-${event.sequence}`,number:calls.length+1,sequence:event.sequence,source,target,phase:event.stage?.startsWith('review.')?event.stage.slice(7):phase,operation,timestamp:event.timestamp,candidate:event.candidate_id});
  }
  const groups=new Map();
  for(const call of calls){const id=`${call.source}:${call.target}`;if(!groups.has(id))groups.set(id,{id,source:call.source,target:call.target,calls:[]});groups.get(id).calls.push(call);}
  return {calls,connections:[...groups.values()]};
}
export const summaries={
  ingestion:['Loaded the complete official dataset.','Preserved the three official partitions.','Confirmed nine classes and RGB input.'],
  profiling:['Profiled class balance and image quality.','Identified potentially confusable tissues.','Found no duplicate or blank training images.'],
  prior_art:['Used three curated sources to form hypotheses.','Proposed local tokenization and compact attention.','Kept new architecture proposals separate from built-in operations.'],
  preprocessing:['Standardized RGB with training-only statistics.','Used six train-only augmentation operations.','Kept preprocessing fixed across the model search.'],
  data_audit:['Verified indices, labels, pixels and shapes.','Confirmed zero train–validation image overlap.','Cleared all eleven pre-training checks.'],
  architecture_research:['Prioritized a compact convolution-tokenizer transformer.','Included multiscale attention and ResNet18 for comparison.','Matched candidates to 28 × 28 inputs.'],
  model_search:['Compared five transformers and three ResNet18 variants.','Selected tuned ResNet18 using validation only.','Saved a checkpoint with 99.59% validation accuracy.'],
  training:['Used AdamW and a cosine learning-rate schedule.','Retained the epoch-17 ResNet18 checkpoint.','Completed 29 epochs with early stopping.'],
  evaluation:['Evaluated all 7,180 held-out test images.','Measured 91.78% accuracy and 0.8900 macro F1.','Compared precision and recall across all nine tissues.'],
  abstention:['Calibrated entropy thresholds on validation.','Achieved 97.59% accuracy on accepted test images.','Routed 2,206 uncertain images to human review.'],
  reporting:['Summarized model selection and test performance.','Clarified the final CNN architecture.','Incorporated the reviewer’s corrections.'],
  orchestrator:['Coordinated eleven task phases and their reviews.','Preserved the validation-selected configuration.','Completed the run and consolidated its findings.'],
  reviewer:['Reviewed each task against its available evidence.','Checked metrics, selection and scientific claims.','Helped refine the final report.'],
  worker:['Executed 32 verification, training and prediction jobs.','Trained all eight candidate models.','Produced predictions for evaluation and uncertainty analysis.']
};
export const phasePositions={ingestion:{x:230,y:120},profiling:{x:450,y:120},prior_art:{x:670,y:120},preprocessing:{x:890,y:120},data_audit:{x:230,y:260},architecture_research:{x:450,y:260},model_search:{x:670,y:260},training:{x:890,y:260},evaluation:{x:230,y:400},abstention:{x:450,y:400},reporting:{x:670,y:400},orchestrator:{x:0,y:260},reviewer:{x:230,y:0},worker:{x:890,y:0}};
