# AlphaZero-Style-Go-Bot

*insert line numbers
some of it is very messy sorry about that, future projects will be better
can customize runtime
can choose if you want it to keep running until confident(for 'medium' settings it will be right most of the time, 'tuned' is supposed to be right 99.7% of the time
can choose if you want basic strategy to be used in rollouts or not
can change game rules
can change if you want to print extra info

low gets most hands right, medium gets most splits right, tuned is supposed to get 99.7% right
go for medium for non-splits, tuned is built so that it is very careful so it doesn't make any mistakes
my computer took 5 min for 10 mil itters so 33k itters/sec
most hands will take about 300,000 to 1 mil itters on medium settings, but splits will take ~6x longer
the hardest hands that have very close decisions will take possibly 30 million itterations 
if you aren't on tuned settings there could be errors in splits and ev will be less accurate (especially for splits)


Try these out!
pair 10 vs 6: tc 4 (5 min runtime)
hard 8 vs 6: tc 2
hard 16 vs 10: tc 1
hard 12 vs 2: tc 3
soft 19 vs 6: tc -1 (133 sec)


feel free to tweak other settings although their effects may be more conplicated than you think
