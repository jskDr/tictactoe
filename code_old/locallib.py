def input_default_with(str, defalut_value, dtype=int):
    answer = input(str+f"[default={defalut_value}] ")
    if answer == '':
        return defalut_value
    else:
        return dtype(answer)