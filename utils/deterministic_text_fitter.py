import os

import PIL.Image
from PIL import ImageDraw, ImageFont

from outer.colors_tools import *
from utils.bboxes import BBox
from utils.glyphs_font_checker import font_supports_all_glyphs
from utils.image_clustering import cluster
from utils.text_fitter import binary_search


def place_textbox_in_rect(rect, textbox, full=False):
    f_w, f_h = textbox
    y_top, x_left, y_bottom, x_right = rect
    rect_width = x_right - x_left
    rect_height = y_bottom - y_top
    true_rect_center = (x_left + rect_width / 2, y_top + rect_height / 2)
    if not full:
        return int(true_rect_center[0] - f_w / 2), int(true_rect_center[1] - f_h / 2)
    return int(true_rect_center[0] - f_w / 2), int(true_rect_center[1] - f_h / 2), \
           int(true_rect_center[0] + f_w / 2), int(true_rect_center[1] + f_h / 2)


def find_two_biggest_not_overlapped_boxes(sorted_lst):
    first = sorted_lst[0]
    first_b_box = BBox(*first[1:5])
    for lst in sorted_lst:
        center, x_left, y_top, x_right, y_bottom, area = lst
        cur_b_box = BBox(x_left, y_top, x_right, y_bottom)
        if not cur_b_box.overlaps(first_b_box):
            return first, lst
    return first, None


def find_two_biggest_specified_not_overlapped_boxes(sorted_lst, is_horizontal_1, is_horizontal_2):
    horizontals = []
    verticals = []
    for lst in sorted_lst:
        if is_horizontal_lst(lst):
            horizontals.append(lst)
        else:
            verticals.append(lst)
    if is_horizontal_1:
        l1 = horizontals
    else:
        l1 = verticals
    if len(l1) == 0:
        return None, None
    first = l1[0]
    first_b_box = BBox(*first[1:5])
    if is_horizontal_2:
        l2 = horizontals
    else:
        l2 = verticals
    for lst in l2:
        center, x_left, y_top, x_right, y_bottom, area = lst
        cur_b_box = BBox(x_left, y_top, x_right, y_bottom)
        if not cur_b_box.overlaps(first_b_box):
            return first, lst
    return first, None


def get_best_font(phrase_words, fit_func, fonts_dir, font_size_boundaries=(8, 200), for_svg=False):
    max_tries = 50
    for test_ind in range(max_tries):
        try:
            font_name, font_path = get_random_font(fonts_dir, for_svg=for_svg)
            if test_ind < max_tries - 1 and not font_supports_all_glyphs(phrase_words, font_path):
                continue

            def bs_fit_predicate(size):
                return fit_func(ImageFont.truetype(font_path, size, encoding="utf-8"))

            lower_bound = font_size_boundaries[0]
            fontsize = binary_search(bs_fit_predicate, *font_size_boundaries)
            if fontsize is None:
                if bs_fit_predicate(lower_bound):
                    fontsize = lower_bound
                else:
                    fontsize = lower_bound - 1
            font = ImageFont.truetype(font_path, fontsize, encoding="utf-8")
            # print(f"Font `{font_name}` with fontsize={fontsize}pt chosen.")
            return font, font_name, fontsize
        except:
            continue


def paste_horizontal_text(lst, phrase, segmented_image, fonts_dir, for_svg=False, debug=False):
    _, x_left, y_top, x_right, y_bottom, area = lst
    rect_height = x_right - x_left
    rect_width = y_bottom - y_top

    def fit_func(font):
        f_w, f_h = font.getsize(phrase)
        return f_h < rect_height * 0.95 and f_w < rect_width * 0.95

    font, font_name, fontsize = get_best_font([phrase], fit_func, fonts_dir, for_svg=for_svg)
    text_x_left, text_y_top, text_x_right, text_y_bottom = place_textbox_in_rect([x_left, y_top, x_right, y_bottom],
                                                                                 font.getsize(phrase), full=True)
    center_color = get_color(segmented_image, text_x_left, text_x_right, text_y_bottom, text_y_top)
    ascent, descent = font.getmetrics()
    result = [{"type": "text",
               "text_xy_left_top": (text_x_left, text_y_top),
               "text_xy_right_bottom": (text_x_right, text_y_bottom),
               "ascent": ascent,
               "descent": descent,
               "word": phrase,
               "color": center_color,
               "font": font}]
    if debug:
        result.append({"type": "circle",
                       "r": 2,
                       "text_xy_left_top": (text_x_left, text_y_top),
                       "xy": (text_x_left, text_y_bottom),
                       "ascent": ascent,
                       "descent": descent,
                       "color": (255, 0, 0, 255)})
    return result


def get_color(segmented_image, text_x_left, text_x_right, text_y_bottom, text_y_top):
    try:
        center_rgb = segmented_image[(text_y_top + text_y_bottom) // 2][(text_x_left + text_x_right) // 2]
    except:
        center_rgb = segmented_image[text_y_top // 2][text_x_left // 2]
    contrast_rgb = contrast_color(*center_rgb)
    center_color = to_int_color((*contrast_rgb,))
    return center_color


def get_random_font(fonts_dir, for_svg=False):
    while True:
        fnames = os.listdir(fonts_dir)
        if for_svg:
            fnames = list(filter(lambda x: "Condensed" not in x, fnames))
        font_name = np.random.choice(fnames)
        font_path = f"{fonts_dir}/{font_name}"
        try:
            for size in range(5, 100):
                ImageFont.truetype(font_path, size, encoding="utf-8")
        except Exception as e:
            print(f"FONT PATH `{font_path}` went to ERROR!!!")
            print(e)
            continue
        return font_name, font_path


def draw_circle(draw, x, y, r, color):
    def circle_point(y, x, r=5):
        return x - r, y - r, x + r, y + r

    draw.ellipse(circle_point(x, y, r), color)


def draw_rect_by_points(draw, x_left, y_top, x_right, y_bottom, r=1, color=(255, 0, 0, 255)):
    draw_circle(draw, x_left, y_top, r, color)
    draw_circle(draw, x_right, y_top, r, color)
    draw_circle(draw, x_left, y_bottom, r, color)
    draw_circle(draw, x_right, y_bottom, r, color)


def try_unite_words(words, font, rect_width, rect_height, height_coef):
    new_words = []
    ind = 0
    max_width = rect_width * 0.95
    while ind < len(words):
        if font.getsize(words[ind])[0] > max_width:
            # impossible
            return None
        cur_words = []
        while ind < len(words) and font.getsize(" ".join(cur_words))[0] <= max_width:
            maybe_new_word = " ".join(cur_words + [words[ind]])
            if font.getsize(maybe_new_word)[0] < max_width:
                cur_words.append(words[ind])
                ind += 1
            else:
                break
        new_words.append(" ".join(cur_words))
    f_h = sum([font.getsize(word)[1] for word in new_words]) * height_coef
    if f_h < rect_height:
        return new_words
    return None


def paste_vertical_text_by_words(lst, words, segmented_image, fonts_dir, use_word_joining, for_svg=False, debug=False):
    result = []
    _, x_left, y_top, x_right, y_bottom, area = lst
    rect_height = x_right - x_left
    rect_width = y_bottom - y_top
    height_coef = 1.1

    def fit_func(font):
        if not use_word_joining:
            f_h = sum([font.getsize(word)[1] for word in words]) * height_coef
            f_w = max([font.getsize(word)[0] for word in words])
            return f_h < rect_height and f_w < rect_width * 0.95
        new_words = try_unite_words(words, font, rect_width, rect_height, height_coef)
        if new_words is None:
            return False
        return True

    font, font_name, fontsize = get_best_font(words, fit_func, fonts_dir, for_svg=for_svg)
    if use_word_joining:
        new_words = try_unite_words(words, font, rect_width, rect_height, height_coef)
        if new_words is not None:
            words = new_words
    if debug:
        result.append({"type": "rect",
                       "rect_points": (x_left, y_top, x_right, y_bottom),
                       "color": None})
    f_w = max([font.getsize(word)[0] for word in words])
    f_h = sum([font.getsize(word)[1] for word in words]) * height_coef
    full_words_textbox = (f_w, f_h)
    full_text_left, full_text_top = place_textbox_in_rect([x_left, y_top, x_right, y_bottom], full_words_textbox)
    full_text_right = full_text_left + f_w
    cur_top_level = full_text_top
    for ind, word in enumerate(words):
        if ind % 2 == 0:
            color = (255, 0, 0, 255)
        else:
            color = (0, 255, 0, 255)
        f_w, f_h = textbox_size = font.getsize(word)
        f_h = height_coef * f_h
        cur_rect = [cur_top_level, full_text_left, cur_top_level + f_h, full_text_right]
        cur_top_level += f_h
        if debug:
            result.append({"type": "rect",
                           "rect_points": cur_rect,
                           "color": color})
        text_x_left, text_y_top, text_x_right, text_y_bottom = place_textbox_in_rect(cur_rect, textbox_size, full=True)
        center_color = get_color(segmented_image, text_x_left, text_x_right, text_y_bottom, text_y_top)
        ascent, descent = font.getmetrics()
        result.append({"type": "text",
                       "text_xy_left_top": (text_x_left, text_y_top),
                       "text_xy_right_bottom": (text_x_right, text_y_bottom),
                       "word": word,
                       "color": center_color,
                       "ascent": ascent,
                       "descent": descent,
                       "font": font})
        if debug:
            result.append({"type": "circle",
                           "r": 2,
                           "xy": (text_x_left, text_y_bottom),
                           "text_xy_left_top": (text_x_left, text_y_top),
                           "ascent": ascent,
                           "descent": descent,
                           "color": (255, 0, 0, 255)})
    return result


def is_horizontal_lst(lst):
    center, x_left, y_top, x_right, y_bottom, area = lst
    rect_height = x_right - x_left
    rect_width = y_bottom - y_top
    return is_horizontal(rect_width, rect_height)


def is_horizontal(width, height):
    return height < 1.7 * width


def paste_text(lst, phrase, segmented_image, fonts_dir, for_svg=False, debug=False):
    if is_horizontal_lst(lst):
        return paste_horizontal_text(lst, phrase, segmented_image, fonts_dir, for_svg=for_svg, debug=debug)
    if " " in phrase:
        is_letters = False
        words = phrase.strip().split()
    else:
        is_letters = True
        words = phrase
    return paste_vertical_text_by_words(lst, words, segmented_image, fonts_dir,
                                        use_word_joining=not is_letters,
                                        for_svg=for_svg, debug=debug)


def check_width_and_height(result, min_width, min_height):
    for to_draw in result:
        if to_draw["type"] == "text":
            font = to_draw["font"]
            word = to_draw["word"]
            w, h = font.getsize(word)
            if h < min_height or w < min_width * len(word):
                return False
    return True


def get_all_boxes_info_to_paste(artist_name, track_name, image, fonts_dir, for_svg=False):
    print(f"Drawing phrases for track `{artist_name} - {track_name}`...")
    width, height, channels = image.shape
    w_cnt, h_cnt = (15, 20)
    ks = [7, 6, 5, 4, 3, 2]
    full_phrase = f"{artist_name} – {track_name}"
    min_text_width = lambda symbols_cnt: width * 0.02 * symbols_cnt
    min_text_height = height * 0.03

    for k in ks:
        print(f"Trying k = {k} ...")
        segmented_image, labels, centers = cluster(image, k, with_labels_centers=True)
        centers_gray = np.uint8([i * 128 / (k - 1) for i in range(k)])
        gray_segm = centers_gray[labels.flatten()].reshape((width, height))
        points = [((w_num + 0.5) * width / w_cnt,
                   (h_num + 0.5) * height / h_cnt)
                  for w_num in range(w_cnt)
                  for h_num in range(h_cnt)]
        results = []
        for x, y in points:
            x = int(x)
            y = int(y)
            base_val = gray_segm[x][y]
            x_left, y_top, x_right, y_bottom = find_best_box(base_val, gray_segm, height, width, x, y)
            results.append([(x, y), x_left, y_top, x_right, y_bottom,
                            (x_right - x_left) * (y_bottom - y_top)])

        def filter_func(lst, symbols_cnt):
            center, x_left, y_top, x_right, y_bottom, area = lst
            rect_height = x_right - x_left
            rect_width = y_bottom - y_top
            return rect_width >= min_text_width(symbols_cnt) and rect_height >= min_text_height

        # results.sort(key=lambda lst: lst[4] - lst[2], reverse=True)
        results.sort(key=lambda lst: lst[5], reverse=True)

        # results_for_full_phrase = list(filter(lambda lst: filter_func(lst, len(full_phrase)), results))
        # if len(results_for_full_phrase) != 0:
        #     return paste_text(results_for_full_phrase[0], full_phrase, segmented_image, for_svg=for_svg)

        def check_sizes(res):
            return check_width_and_height(res, width / 35, height / 35)

        # Type 1: horizontally `artist_name - track_name`
        for lst in results:
            if is_horizontal_lst(lst):
                result = paste_horizontal_text(lst, full_phrase, segmented_image, fonts_dir, for_svg=for_svg)
                if check_sizes(result):
                    return result
                break
        # Type 2: vertically `artist_name - track_name` by letters
        for lst in results:
            if not is_horizontal_lst(lst):
                result = paste_vertical_text_by_words(lst, full_phrase, segmented_image, fonts_dir,
                                                      use_word_joining=False, for_svg=for_svg)
                if check_sizes(result):
                    return result
                break
        # Type 4: two biggest not overlapped boxes
        box_a, box_b = ans = find_two_biggest_not_overlapped_boxes(results)
        if None in ans:
            continue
        fst, snd = artist_name, track_name
        if len(artist_name) < len(track_name):
            fst, snd = track_name, artist_name
        result = []

        def try_many_variants(lst, phrase):
            if not is_horizontal_lst(lst):
                # try by letters
                res = paste_vertical_text_by_words(lst, phrase, segmented_image, fonts_dir,
                                                   use_word_joining=False, for_svg=for_svg)
                if not check_sizes(res):
                    if " " in phrase:
                        # try by words
                        res = paste_vertical_text_by_words(lst, phrase.split(), segmented_image, fonts_dir,
                                                           use_word_joining=False, for_svg=for_svg)
                        if not check_sizes(res):
                            # try by words with joining
                            res = paste_vertical_text_by_words(lst, phrase.split(), segmented_image, fonts_dir,
                                                               use_word_joining=True, for_svg=for_svg)
            else:
                res = paste_horizontal_text(lst, phrase, segmented_image, fonts_dir, for_svg=for_svg)
            return res

        result.extend(try_many_variants(box_a, fst))
        result.extend(try_many_variants(box_b, snd))
        if check_sizes(result):
            return result
        # Type 3: vertically `artist_name` and `track_name` as two words in one box
        with_biggest_area = list(filter(lambda lst: lst[5] * 2 > results[0][5], results))
        with_biggest_area.sort(key=lambda lst: lst[4] - lst[2], reverse=True)
        for lst in with_biggest_area:
            result = paste_vertical_text_by_words(lst, [artist_name, track_name], segmented_image, fonts_dir,
                                                  use_word_joining=False, for_svg=for_svg)
            if check_sizes(result):
                return result
            break
        if k != ks[-1] and not check_sizes(result):
            continue
        print("Using the very basic variant for font drawing.")
        padding_coef = 20
        lst1 = 0, width // padding_coef, height // padding_coef, width - width // padding_coef, height // 2, 0
        lst2 = 0, width // padding_coef, height // 2, width, height - height // padding_coef, 0
        result = []
        result.extend(paste_text(lst1, artist_name, segmented_image, fonts_dir, for_svg=for_svg))
        result.extend(paste_text(lst2, track_name, segmented_image, fonts_dir, for_svg=for_svg))
        return result


def draw_phrases(artist_name, track_name, img_path, output_path, fonts_dir):
    pil_image = PIL.Image.open(img_path).convert(mode='RGB')
    draw = ImageDraw.Draw(pil_image)
    to_draw = get_all_boxes_info_to_paste(artist_name, track_name, np.asarray(pil_image.copy()), fonts_dir)
    for x in to_draw:
        draw_to_draw_object(draw, x)
    pil_image.save(output_path)


def draw_to_draw_object(draw, d):
    if d["type"] == "rect":
        if d["color"] is None:
            draw_rect_by_points(draw, *(d["rect_points"]))
        else:
            draw_rect_by_points(draw, *(d["rect_points"]), color=d["color"])
    elif d["type"] == "text":
        draw.text(d["text_xy_left_top"], d["word"], fill=d["color"], font=d["font"])
    elif d["type"] == "circle":
        p_y, p_x = d["xy"]
        r = d["r"]
        descent = d["descent"]
        ascent = d["ascent"]
        y_t, x_l = d["text_xy_left_top"]
        # draw.ellipse(circle_point(x_l, y_t, 1), d["color"])
        # draw.ellipse(circle_point(p_x - descent, p_y, 2), d["color"])
        draw_circle(draw, x_l + ascent, y_t, 2, d["color"])


def find_best_box(base_val, gray_segm, height, width, x, y):
    x_left, y_top, x_right, y_bottom = x, y, x, y
    x_left_done = y_top_done = x_right_done = y_bottom_done = False
    back_coef = int(width / 50)

    def check_value(row, value):
        return len(row) == 0 or min(row) == max(row) == value

    while not (x_left_done and y_top_done and x_right_done and y_bottom_done):
        if not x_left_done:
            if 0 < x_left < width - 1 and check_value(gray_segm[x_left - 1, y_top:y_bottom], base_val):
                x_left -= 1
            else:
                x_left += back_coef
                x_left_done = True
        if not y_top_done:
            if 0 < y_top < height - 1 and check_value(gray_segm[x_left:x_right, y_top - 1], base_val):
                y_top -= 1
            else:
                y_top += back_coef
                y_top_done = True
        if not x_right_done:
            if 0 < x_right < width - 1 and check_value(gray_segm[x_right + 1, y_top:y_bottom], base_val):
                x_right += 1
            else:
                x_right -= back_coef
                x_right_done = True
        if not y_bottom_done:
            if 0 < y_bottom < height - 1 and check_value(gray_segm[x_left:x_right, y_bottom + 1], base_val):
                y_bottom += 1
            else:
                y_bottom -= back_coef
                y_bottom_done = True
    return x_left - back_coef, y_top - back_coef, x_right + back_coef, y_bottom + back_coef,
