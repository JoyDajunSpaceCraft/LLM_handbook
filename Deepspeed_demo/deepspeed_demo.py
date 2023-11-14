import deepspeed

# initial Deepspeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=ds_args,  # deepspeed
    model=model,
    model_parameters=model.parameters(),
    config_params='ds_config.json'
)

for epoch in range(num_epochs):
    for batch in data_loader:
        # move the data to the device
        input_ids = batch['input_ids'].to(model_engine.local_rank)
        attention_mask = batch['attention_mask'].to(model_engine.local_rank)

        # forward
        outputs = model_engine(input_ids, attention_mask=attention_mask)

        # loss
        loss = outputs.loss

        # backward
        model_engine.backward(loss)

        # update 
        model_engine.step()


