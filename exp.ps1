$mechs = @( 'pmsub','topm', 'duchi', 'pm', 'to', 'noutput')
$datasets = @('adult', 'truncNormal', 'employee','hpc_voltage')

foreach ($dataset in $datasets){
    foreach ($mech in $mechs){
            python EXP.py --dataset $dataset --mech $mech --alpha 0.05 --lr 0.3 --steps 30 --tau 2 --zeta 0.3 --eps_ratio 0.7
    }
}