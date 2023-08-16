#! /usr/bin/env python

import ROOT

rfile = ROOT.TFile("DY_WJets_QCDPt5_combined.root")
peta = rfile.Get("peta")

nbx = peta.GetNbinsX()
pmax = peta.GetXaxis().GetBinLowEdge(nbx) + peta.GetXaxis().GetBinWidth(nbx)

pt = ROOT.TH1F("pt","pt",nbx,0,pmax)

for i in range(0,nbx+2):
    # pt.SetBinContent(i, max((peta.GetBinContent(i,1)+peta.GetBinContent(i,1))/2, 0))   # eta=0.0
    # pt.SetBinContent(i, max((peta.GetBinContent(i,24)+peta.GetBinContent(i,25))/2, 0))   # eta=0.6
    pt.SetBinContent(i, max((peta.GetBinContent(i,7)+peta.GetBinContent(i,7))/2, 0))   # eta=0.16
    pt.SetBinError(i, peta.GetBinError(i,1))

# kill bins with pt too low to make it to detector (saves running time)
# each bin has width of 2.5 GeV
pt.SetBinContent(1,0)
pt.SetBinContent(2,0)
pt.SetBinContent(3,0)
pt.SetBinContent(4,0)
pt.SetBinContent(5,0)

# outfile = ROOT.TFile("combined_PtSpect_Eta0p0.root","RECREATE")
# outfile = ROOT.TFile("combined_Pt15Spect_Eta0p6.root","RECREATE")
outfile = ROOT.TFile("combined_PtSpect_Eta0p16.root","RECREATE")
pt.Write()
outfile.Close()








