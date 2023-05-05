import numpy as np
import pandas as pd
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from propy import AAComposition, CTD, Autocorrelation, QuasiSequenceOrder, PyPro
from tqdm import tqdm


def pc_properties(sequences, target):
    vocab = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }

    selected_seqs = []
    selected_tgt = []
    sequences = np.array(sequences)
    target = np.array(target)

    # remove the peptides with uncommon amino acid.
    for i, s in enumerate(sequences):
        flag = False
        if s.find(',') != -1:
            continue
        for w in s:
            if w not in vocab.keys():
                flag = True
        if not flag:
            selected_seqs.append(s)
            selected_tgt.append(target[i])
    properties = []
    # feature's name
    col = ['Eisenberg hydrophobicity', 'Eisenberg hydrophobicity monment', 'GRAVY hydrophobicity',
           'GRAVY hydrophobic moment', 'z3-1',
           'z3-2', 'z3-3', 'z5-1', 'z5-2', 'z5-3', 'z5-4', 'z5-5',
           'AASI index', 'AASI index moment', 'ABHPRK', 'argos index',
           'argos index moment', 'bulkiness index', 'bulkiness index moment', 'charge_phys index', 'charge_acid index',
           'Ez', 'flexibility scale',
           'flexibility scale monent', 'grantham',
           'Hopp-Woods hydrophobicity scale', 'Hopp-Woods hydrophobicity moment', 'ISAECI',
           'Janin hydrophobicity scale', 'Janin hydrophobicity moment',
           'Kyte & Doolittle hydrophobicity scale', 'Kyte & Doolittle hydrophobicity moment',
           'Levitt alpha-helix propensity scale',
           'Levitt alpha-helix propensity moment', 'MSS scale', 'MSS monment', 'MSW', 'pepArc', 'pepcats',
           'AA polarity', 'AA polarity moment',
           'PPCALI', 'AA refractivity', 'AA refractivity moment', 't_scale', 'TM_tend_scale', 'TM_tend_monment',
           'sequence length', 'Boman index',
           'global aromaticity', 'aliphatic index', 'instability index', 'net charge', 'molecular weight',
           'isoelectric point', 'hydrophobic_ratio']

    for k in AAComposition.CalculateAAComposition(sequences[0]).keys():
        col.append(k)
    for k in AAComposition.CalculateDipeptideComposition(sequences[0]).keys():
        col.append(k)
    for k in AAComposition.GetSpectrumDict(sequences[0]).keys():
        col.append(k)
    for k in CTD.CalculateCTD(sequences[0]).keys():
        col.append(k)
    for k in Autocorrelation.CalculateGearyAutoTotal(sequences[0]).keys():
        col.append(k)
    for k in Autocorrelation.CalculateMoranAutoTotal(sequences[0]).keys():
        col.append(k)
    for k in Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(sequences[0]).keys():
        col.append(k)
    for k in QuasiSequenceOrder.GetQuasiSequenceOrder2Grant(sequences[0], 50).keys():
        col.append(k)
    for k in QuasiSequenceOrder.GetQuasiSequenceOrder2SW(sequences[0], 50).keys():
        col.append(k)
    for k in PyPro.GetProDes(sequences[0]).GetPAAC(lamda=1, weight=0.05).keys():
        col.append(k)
    for k in QuasiSequenceOrder.GetSequenceOrderCouplingNumberGrant(sequences[0], 45).keys():
        col.append(k)
    for k in QuasiSequenceOrder.GetSequenceOrderCouplingNumberSW(sequences[0], 45).keys():
        col.append(k)

    for i, s in tqdm(enumerate(sequences)):
        pepdesc = PeptideDescriptor(s)
        pepdesc.load_scale('eisenberg')
        pepdesc.calculate_global()  # calculate global Eisenberg hydrophobicity
        pepdesc.calculate_moment(append=True)

        pepdesc.load_scale('gravy')  # load GRAVY scale
        pepdesc.calculate_global(append=True)  # calculate global GRAVY hydrophobicity
        pepdesc.calculate_moment(append=True)  # calculate GRAVY hydrophobic moment

        pepdesc.load_scale('z3')  # load old Z scale
        pepdesc.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)

        pepdesc.load_scale('z5')  # load old Z scale
        pepdesc.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)

        pepdesc.load_scale('AASI')
        pepdesc.calculate_global(append=True)  # calculate global AASI index
        pepdesc.calculate_moment(append=True)  # calculate AASI index moment

        pepdesc.load_scale('ABHPRK')
        pepdesc.calculate_global(append=True)  # calculate ABHPRK feature

        pepdesc.load_scale('argos')
        pepdesc.calculate_global(append=True)  # calculate global argos index
        pepdesc.calculate_moment(append=True)  # calculate argos index moment

        pepdesc.load_scale('bulkiness')
        pepdesc.calculate_global(append=True)  # calculate global bulkiness index
        pepdesc.calculate_moment(append=True)  # calculate bulkiness index moment

        pepdesc.load_scale('charge_phys')
        pepdesc.calculate_global(append=True)  # calculate global charge_phys index

        pepdesc.load_scale('charge_acid')
        pepdesc.calculate_global(append=True)  # calculate global charge_acid index

        pepdesc.load_scale('Ez')
        pepdesc.calculate_global(
            append=True)  # calculate global energies of insertion of amino acid side chains into lipid bilayers index

        pepdesc.load_scale('flexibility')
        pepdesc.calculate_global(append=True)  # calculate global flexibility scale
        pepdesc.calculate_moment(append=True)  # calculate flexibility moment

        pepdesc.load_scale('grantham')
        pepdesc.calculate_global(
            append=True)  # calculate global amino acid side chain composition, polarity and molecular volume

        pepdesc.load_scale('hopp-woods')
        pepdesc.calculate_global(append=True)  # calculate global Hopp-Woods hydrophobicity scale
        pepdesc.calculate_moment(append=True)  # calculate Hopp-Woods hydrophobicity moment

        pepdesc.load_scale('ISAECI')
        pepdesc.calculate_global(
            append=True)  # calculate global ISAECI (Isotropic Surface Area (ISA) and Electronic Charge Index (ECI) of amino acid side chains) index

        pepdesc.load_scale('janin')
        pepdesc.calculate_global(append=True)  # calculate global Janin hydrophobicity scale
        pepdesc.calculate_moment(append=True)  # calculate Janin hydrophobicity moment

        pepdesc.load_scale('kytedoolittle')
        pepdesc.calculate_global(append=True)  # calculate global Kyte & Doolittle hydrophobicity scale
        pepdesc.calculate_moment(append=True)  # calculate Kyte & Doolittle hydrophobicity moment

        pepdesc.load_scale('levitt_alpha')
        pepdesc.calculate_global(append=True)  # calculate global Levitt alpha-helix propensity scale
        pepdesc.calculate_moment(append=True)  # calculate Levitt alpha-helix propensity moment

        pepdesc.load_scale('MSS')
        pepdesc.calculate_global(
            append=True)  # calculate global MSS index, graph-theoretical index that reflects topological shape and size of amino acid side chains
        pepdesc.calculate_moment(append=True)  # calculate MSS moment

        pepdesc.load_scale('MSW')
        pepdesc.calculate_global(
            append=True)  # calculate global MSW scale, Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM), extended to natural amino acids

        pepdesc.load_scale('pepArc')
        pepdesc.calculate_global(
            append=True)  # calculate global pepArc, modlabs pharmacophoric feature scale, dimensions are: hydrophobicity, polarity, positive charge, negative charge, proline.

        pepdesc.load_scale('pepcats')
        pepdesc.calculate_global(
            append=True)  # calculate global pepcats, modlabs pharmacophoric feature based PEPCATS scale

        pepdesc.load_scale('polarity')
        pepdesc.calculate_global(append=True)  # calculate global AA polarity
        pepdesc.calculate_moment(append=True)  # calculate AA polarity moment

        pepdesc.load_scale('PPCALI')
        pepdesc.calculate_global(
            append=True)  # calculate global modlabs inhouse scale derived from a PCA of 143 amino acid property scales

        pepdesc.load_scale('refractivity')
        pepdesc.calculate_global(append=True)  # calculate global relative AA refractivity
        pepdesc.calculate_moment(append=True)  # calculate relative AA refractivity moment

        pepdesc.load_scale('t_scale')
        pepdesc.calculate_global(
            append=True)  # calculate global t scale, A PCA derived scale based on amino acid side chain properties calculated with 6 different probes of the GRID program

        pepdesc.load_scale('TM_tend')
        pepdesc.calculate_global(append=True)  # calculate global Amino acid transmembrane propensity scale
        pepdesc.calculate_moment(append=True)

        globdesc = GlobalDescriptor(s)
        globdesc.length()  # sequence length
        globdesc.boman_index(append=True)  # Boman index
        globdesc.aromaticity(append=True)  # global aromaticity
        globdesc.aliphatic_index(append=True)  # aliphatic index
        globdesc.instability_index(append=True)  # instability index
        globdesc.calculate_charge(ph=7.4, amide=False, append=True)  # net charge
        globdesc.calculate_MW(amide=False, append=True)  # molecular weight
        globdesc.isoelectric_point(amide=False, append=True)  # isoelectric point
        globdesc.hydrophobic_ratio(append=True)

        glob = np.concatenate([pepdesc.descriptor, globdesc.descriptor], axis=1)

        aa = []

        for a in AAComposition.CalculateAAComposition(s).values():
            aa.append(a)
        dpc = []
        for d in AAComposition.CalculateDipeptideComposition(s).values():
            dpc.append(d)
        tpc = []
        for t in AAComposition.GetSpectrumDict(s).values():
            tpc.append(t)
        ctd = []
        for c in CTD.CalculateCTD(s).values():
            ctd.append(c)
        ga = []
        for g in Autocorrelation.CalculateGearyAutoTotal(s).values():
            ga.append(g)
        ma = []
        for m in Autocorrelation.CalculateMoranAutoTotal(s).values():
            ma.append(m)
        mba = []
        for mb in Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(s).values():
            mba.append(mb)
        qso = []
        for qs in QuasiSequenceOrder.GetQuasiSequenceOrder2Grant(s, 50).values():
            qso.append(qs)
        for qs in QuasiSequenceOrder.GetQuasiSequenceOrder2SW(s, 50).values():
            qso.append(qs)
        paac = []
        for k, v in PyPro.GetProDes(s).GetPAAC(lamda=1, weight=0.05).items():
            paac.append(v)
        so = []
        for o in QuasiSequenceOrder.GetSequenceOrderCouplingNumberGrant(s, 45).values():
            so.append(o)
        for o in QuasiSequenceOrder.GetSequenceOrderCouplingNumberSW(s, 45).values():
            so.append(o)
        p = np.concatenate([np.array(glob).reshape(1, -1), np.array(aa).reshape(1, -1), np.array(dpc).reshape(1, -1),
                            np.array(tpc).reshape(1, -1), np.array(ctd).reshape(1, -1), np.array(ga).reshape(1, -1),
                            np.array(ma).reshape(1, -1), np.array(mba).reshape(1, -1),
                            np.array(qso).reshape(1, -1), np.array(paac).reshape(1, -1), np.array(so).reshape(1, -1)],
                           axis=1)
        properties.append(p)
    properties = np.array(properties)
    properties = properties.reshape(properties.shape[0], properties.shape[2])
    df = pd.DataFrame(properties, columns=col)
    return np.array(properties), np.array(selected_tgt), df