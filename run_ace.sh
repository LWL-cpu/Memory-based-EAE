for seed in  777
do
    for lr in 2e-5
    do
        bash ./scripts/train_ace_large.sh $seed $lr
    done
done

for seed in  777
do
    for lr in 2e-5
    do
        bash ./scripts/train_wikievent.sh $seed $lr
    done
done
