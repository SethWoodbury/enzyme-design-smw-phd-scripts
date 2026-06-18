# align all loaded structures to the first one of the list 
def align_all_on(sele, aln_type='align'):
    all_objs = cmd.get_object_list()
    all_seles = [cmd.select(f'{x}_alignment_resis',f'model {x} and {sele}') for x in all_objs]
    all_seles = [f'{x}_alignment_resis' for x in all_objs]
    print(all_seles)
    for i in range(1,len(all_objs)):
        print(all_seles[i])
        if aln_type=='cealign':    
            cmd.cealign(all_seles[i], all_seles[0])
        elif aln_type=='super':
            cmd.super(all_seles[i], all_seles[0])
        else:
            cmd.align(all_seles[i], all_seles[0])
    
    cmd.center(all_objs[0], animate=-1)
    [cmd.delete(f'{x}_alignment_resis') for x in all_objs]

cmd.extend('align_all_on',align_all_on)
