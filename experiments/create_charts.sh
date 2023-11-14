cat results_header results_bs1                          |              python summary_chart.py stdin png 1      img_s,memory vit_b,vit_h p4d_charts/bar_chart_0
cat results_header results_bs1 results_bs8              | head -n  5 | python summary_chart.py stdin png 1,8    img_s,memory vit_b,vit_h p4d_charts/bar_chart_1
cat results_header results_bs1 results_bs8              |              python summary_chart.py stdin png 1,8    img_s,memory vit_b,vit_h p4d_charts/bar_chart_2
cat results_header results_bs1 results_bs8 results_bs32 | head -n  9 | python summary_chart.py stdin png 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_3
cat results_header results_bs1 results_bs8 results_bs32 | head -n 11 | python summary_chart.py stdin png 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_4
cat results_header results_bs1 results_bs8 results_bs32 | head -n 13 | python summary_chart.py stdin png 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_5
cat results_header results_bs1 results_bs8 results_bs32 | head -n 15 | python summary_chart.py stdin png 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_6
cat results_header results_bs1 results_bs8 results_bs32 | head -n 17 | python summary_chart.py stdin png 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_7

cat results_header results_bs1                          |              python summary_chart.py stdin svg 1      img_s,memory vit_b,vit_h p4d_charts/bar_chart_0
cat results_header results_bs1 results_bs8              | head -n  5 | python summary_chart.py stdin svg 1,8    img_s,memory vit_b,vit_h p4d_charts/bar_chart_1
cat results_header results_bs1 results_bs8              |              python summary_chart.py stdin svg 1,8    img_s,memory vit_b,vit_h p4d_charts/bar_chart_2
cat results_header results_bs1 results_bs8 results_bs32 | head -n  9 | python summary_chart.py stdin svg 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_3
cat results_header results_bs1 results_bs8 results_bs32 | head -n 11 | python summary_chart.py stdin svg 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_4
cat results_header results_bs1 results_bs8 results_bs32 | head -n 13 | python summary_chart.py stdin svg 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_5
cat results_header results_bs1 results_bs8 results_bs32 | head -n 15 | python summary_chart.py stdin svg 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_6
cat results_header results_bs1 results_bs8 results_bs32 | head -n 17 | python summary_chart.py stdin svg 1,8,32 img_s,memory vit_b,vit_h p4d_charts/bar_chart_7
