import numpy as np
import pygame as pg


def create_matrix(i: int):
    a = np.arange(1, i ** 2 + 1).reshape(i, i)
    # a = np.asarray([x * 2 for x in [x for x in range(1, i * i)]]).reshape(i, i)
    return a


def m_split(_m, _x):
    return [np.vsplit(x, _x) for x in np.hsplit(_m, _x)]


def m_join(_m, _x):
    return np.vstack([np.hstack(np.vstack(_m)[x::_x]) for x in range(_x)])


def m_rot(_m, idxs):
    a0 = _m[idxs[0][0]][idxs[0][1]]
    for i in range(len(idxs) - 1):
        _m[idxs[i][0]][idxs[i][1]] = _m[idxs[i + 1][0]][idxs[i + 1][1]]
    _m[idxs[-1][0]][idxs[-1][1]] = a0
    return _m


def magic_steps(_m, _steps):
    m0 = _m
    for step in _steps:
        m0 = m_rot(m0, step)
    return m0


m_3 = [[[1, 0], [0, 0], [1, 2], [2, 2]], [[2, 0], [2, 1], [0, 2], [0, 1]]]
m_4 = [[[0, 0], [2, 2]], [[1, 0], [0, 1]],
       [[2, 0], [3, 1]], [[3, 0], [1, 2]],
       [[0, 2], [1, 3]], [[1, 1], [3, 3]],
       [[2, 1], [0, 3]], [[3, 2], [2, 3]]]

m_4_1 = [[[1, 0], [2, 3]], [[2, 0], [1, 3]], [[0, 1], [3, 2]], [[0, 2], [3, 1]]]

m_5 = [[[0, 0], [2, 3], [1, 3], [0, 2]], [[1, 2], [1, 1], [2, 0], [4, 0]],
       [[2, 1], [3, 1], [4, 2], [4, 4]], [[3, 2], [3, 3], [2, 4], [0, 4]],
       [[1, 0], [3, 4]], [[0, 3], [4, 1]], [[3, 0], [0, 1], [1, 4], [4, 3]]]


def pow_square(_root, _pow, _mss):
    m_dmag = lambda x, d: m_join(magic_steps(m_split(x, d), _mss), d)
    _p = _root ** _pow
    m0 = m_dmag(create_matrix(_p), _root)
    for i in range(1, _pow):
        m0 = m_join([[m_dmag(x, _root) for x in y] for y in m_split(m0, _root ** i)], _root ** i)
    return m0


def flip_swap(_m, i):
    _m[[i, len(_m) - i - 1]] = _m[[len(_m) - i - 1, i]]
    _m[:, [i, len(_m) - i - 1]] = _m[:, [len(_m) - i - 1, i]]
    return _m


def to_csv(_m):
    for x in _m:
        print(''.join([str(y) + ';' for y in x]))


def get_cycles(_m):
    ret = []
    dim = len(_m)
    used = []
    j_to_xy = lambda _j, _d: [(_j - 1) % _d, int((_j - 1) / _d)]
    xy_to_j = lambda _xy, _d: _xy[0] + _xy[1] * _d + 1
    for j in range(1, dim * dim + 1):
        _p = []
        x, y = j_to_xy(j, dim)
        if _m[y][x] in used:
            continue
        if _m[y][x] == j:
            used += [_m[y][x]]
            continue
        _p += [[x, y]]
        while True:
            a = _m[y][x]
            x, y = j_to_xy(a, dim)
            _p += [[x, y]]
            used += [a]
            if xy_to_j([x, y], dim) == j:
                ret += [_p]
                break
    return ret


def get_statics(_m):
    dim = len(_m)
    j_to_xy = lambda _j, _d: [(_j - 1) % _d, int((_j - 1) / _d)]
    ret = []
    for j in range(1, dim * dim + 1):
        x, y = j_to_xy(j, dim)
        if _m[y][x] == j:
            ret += [[x, y]]
    return ret


def to_star(_m):
    dim = len(_m)
    max_sz = 1000
    _sz = int(max_sz / dim)
    sz = 20 if _sz > 20 else 1 if _sz == 0 else _sz
    pygame.init()
    d_sz = (sz * dim + 20, sz * dim + 20)
    _sc = pygame.display.set_mode(d_sz)
    _sc.fill((255, 255, 255))
    pg.display.flip()
    for _c in get_cycles(_m):
        sc = _sc.convert_alpha()
        sc.fill([0, 0, 0, 0])
        points = [(x * sz + 10, y * sz + 10) for x, y in _c[:-1]]
        r = pg.draw.lines(sc, (0, 0, 0, 15), True, points) \
            if len(points) > 2 \
            else pg.draw.line(sc, (0, 0, 0, 55), points[0], points[1])
        _sc.blit(sc, (0, 0))
        pg.display.update(r)
    for _s in [(x * sz + 10, y * sz + 10) for x, y in get_statics(_m)]:
        r = pg.draw.circle(_sc, (200, 200, 0, 70), _s, 5)
        pg.display.update(r)
    c = pg.time.Clock()
    while True:
        c.tick(60)
        for events in pg.event.get():
            if events.type == pg.QUIT:
                pg.quit()
        pygame.display.update()


def m_swaps(_m, _x, _root, _pow):
    for _i in [x for x in range(1, int(_root ** pow / 2))][::_x]:
        _m = flip_swap(_m, _i)
    return _m


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pow = 4
    root = 4
    m1 = pow_square(root, pow, m_4_1)
    for i in [3, 5, 2, 4, 6, 7, 12, 27]:
        # swapping columns and rows (chaotic). Comment lines below and above to disable.
        m1 = m_swaps(m1, i, root, pow)
    try:
        to_star(m1)
    except pygame.error:
        print('Drawing finished.')
    print(m1)
    m2 = np.rot90(m1)
    print([sum(x) for x in m1])
    print([sum(x) for x in m2])
    print(sum([m1[x, x] for x in range(root ** pow)]))
    print(sum([m2[x, x] for x in range(root ** pow)]))
