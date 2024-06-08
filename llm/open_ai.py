import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7

client = OpenAI()


def complete_text(prompt):
    res = client.chat.completions.create(
        messages=prompt,
        model=MODEL,
        temperature=TEMPERATURE,
    )
    return res.choices[0].message.content


print(complete_text([
    {"role": "system",
     "content": "The code smell known as Columns and DataType Not Explicitly Set highlights the importance of explicitly selecting columns and setting their data types when importing data, to avoid unexpected behavior in subsequent data processing steps. The problem with not explicitly setting columns and data types during data import is that it can lead to confusion and errors in the downstream data schema. When columns are not explicitly selected, developers may be unsure of the data structure. Similarly, if data types are not set explicitly, the default type conversion might silently pass unexpected inputs, causing errors later in the processing pipeline. To address Columns and DataType Not Explicitly Set, it is recommended to explicitly specify the columns and their data types when importing data. This practice helps in maintaining a clear data schema and ensures that the data types are correctly assigned, thus preventing unexpected behavior and potential errors in downstream tasks"},
    {"role": "system",
     "content": "def forecast(tmp_df, train, index_forecast, days_in_future):\n    \n    # Fit model with training data\n    model = auto_arima(train, trace=False, error_action='ignore', suppress_warnings=True)\n    model_fit = model.fit(train)\n        \n    forecast, confint = model_fit.predict(n_periods=len(index_forecast), return_conf_int=True)\n\n    forecast_df = pd.concat([tmp_df, pd.DataFrame(forecast, index = index_forecast, columns=['pred'])], axis=1, sort=False)\n    date_range = [d.strftime('%Y-%m-%d') for d in pd.date_range(train_start, forecast_end)]\n    forecast_df['date'] = pd.Series(date_range).astype(str)\n    forecast_df[''] = None # Dates get messed up, so need to use pandas plotting\n        \n    # Save Model and file\n    print('... saving file:', forecast_file)\n    forecast_df.to_csv(os.path.join(data_dir, forecast_file))\n        \n    plot_forecast(forecast_df, train, index_forecast, forecast, confint)\n"},
    {"role": "system", "content": "You will be provided with the explanation of a code smell and the body of a function. Your task is to suggest how to fix the code smell provided in the function provided"},
    {"role": "system", "content": "write only the code"},
    {"role": "user",
     "content": "Suggest me how to fix the code smell Columns and DataType Not Explicitly Set in the function forecast"},
]))
