# vaik-mnist-classification-dataset

Create Pascal VOC formatted MNIST classification dataset

## Example

![vaik-mnist-classification-dataset](https://user-images.githubusercontent.com/116471878/200166732-cdc0b74f-11ca-41f2-a25d-974ce81337d2.png)

## Usage

```shell
pip install -r requirements.txt
python main.py --output_dir_path ~/.vaik-mnist-detection-dataset \
                --train_sample_num 10000 \
                --valid_sample_num 100 \
                --char_max_size 128 \
                --char_min_size 64 \
```

## Output
- classes.txt

```text
zero
one
two
three
four
five
six
seven
eight
nine
```

- jpg per classes

![vaik-mnist-classification-dataset-output](https://user-images.githubusercontent.com/116471878/200166812-f963ac85-345f-428a-b2f3-c7fb50eb2f32.png)
