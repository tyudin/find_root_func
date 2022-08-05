import math
from typing import Callable, Union
import numpy as np
from matplotlib import pyplot as plt


def draw_func(func: Callable, left_bound: float, right_bound: float, num_val: int = 50) -> None:
    x = np.linspace(left_bound, right_bound, num_val)
    y = [func(t) for t in x]
    y1 = [0] * num_val

    fig = plt.figure()
    fig.set_size_inches(15, 15)

    plt.plot(x, y, label="F(x)",
             color="red",
             linewidth=1,
             linestyle='solid')
    plt.plot(x, y1, label="y = 0",
             color="blue",
             linewidth=1,
             linestyle='solid')

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend(loc='best', fontsize=14)
    plt.show()


def found_root(func: Callable, left_bound: float, right_bound: float, accuracy: float = 1.0e-5) -> Union[float, None]:
    """ Поиск корня функции методом половинного деления отрезка
    c точностью accuracy (по-умолчанию: 0.00001 """
    assert right_bound > left_bound, "Левая граница больше чем правая!"
    assert accuracy > 0, "Точность должна быть больше нуля!"

    while (right_bound - left_bound) > 2 * accuracy:
        middle = (left_bound + right_bound) / 2

        func_value = func(middle)  # значение функции в середине отрезка
        left_sign = math.copysign(1, func(left_bound))  # знак функции на левом конце
        right_sign = math.copysign(1, func(right_bound))  # знак функции на правом конце

        if func_value == 0:
            # невероятно, но попали в корень!
            return middle

        if func_value > 0:
            if left_sign > 0:
                # функция была положительная
                left_bound = middle
            else:
                # функция была отрицательная
                right_bound = middle
        else:
            if right_sign > 0:
                left_bound = middle
            else:
                right_bound = middle

    left_sign = math.copysign(1, func(left_bound))
    right_sign = math.copysign(1, func(right_bound))

    if left_sign == right_sign:
        # знак функции на границах не меняется => корня нет
        return None

    return (right_bound + left_bound) / 2


def f(x: float) -> float:
    return x * x + 2 * x - 6


if __name__ == '__main__':
    draw_func(f, -5, 5)
    print(f"x1 ~= {found_root(f, -5, 0)} x2 ~= {found_root(f, 0, 5)}")
