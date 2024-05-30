# using yq to edit the config file
yq e ".task.data = \"${data_dir}/${train_data}\" |
.model.w2v_path = \"${w2v_path}\" |
.checkpoint.restore_file = \"${last_chkpt}\" | 
.optimization.max_update = ${max_update}" -i ${hprams_dir}/${hparms}