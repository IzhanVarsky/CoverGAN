from utils.deterministic_text_fitter import draw_phrases

if __name__ == '__main__':
    fonts_dir = r"F:\stazhirovka2021\Diploma\covergan-test\covergan\fonts"


    def draw(*x, **k):
        draw_phrases(*x, **k, fonts_dir=fonts_dir)


    draw("KYUL", "VeryVeryVeryVeryVeryVeryLongWordHere",
         img_path="test_img5.png", output_path="debug_out.png")
    draw("&me", "The Rapture Pt.II", img_path="test_img.png", output_path="sample_output1.png")
    draw("&me", "The Rapture Pt.II", img_path="test_img2.png", output_path="sample_output2.png")
    draw("&me", "The Rapture Pt.II", img_path="test_img3.png", output_path="sample_output3.png")
    draw("&me", "The Rapture Pt.II", img_path="test_img4.png", output_path="sample_output4.png")
    draw("KYUL", "Even Though We Are Not the Same X X X X X XXXX XXXXXX",
         img_path="test_img5.png", output_path="sample_output5.png")
    draw("Кино", "Группа Крови", img_path="test_img5.png", output_path="sample_output6.png")
