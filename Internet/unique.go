package main

import (
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strings"

	"github.com/kshedden/flowtuple"
)

var (
	counts    []int
	udpCounts []int
	tcpCounts []int
	sources   []int
	ports     [][]int
)

func main() {

	fname := os.Args[1]

	counts = make([]int, 60)
	udpCounts = make([]int, 60)
	tcpCounts = make([]int, 60)
	sources = make([]int, 60)
	ports = make([][]int, 60)
	for k := 0; k < 60; k++ {
		ports[k] = make([]int, 65536)
	}

	fid, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer fid.Close()
	gid, err := gzip.NewReader(fid)
	if err != nil {
		panic(err)
	}
	defer gid.Close()

	lf, err := os.Create("testlog.txt")
	if err != nil {
		panic(err)
	}
	logger := log.New(lf, "", log.Ltime)

	ftr := flowtuple.NewFlowtupleReader(gid).SetLogger(logger)

	var frec flowtuple.FlowRec

	for {
		err := ftr.ReadIntervalHead()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}

		sourcesSeen := make(map[int]bool)

		for {
			err := ftr.ReadClassHead()
			if err == io.EOF {
				break
			} else if err != nil {
				panic(err)
			}

			for {
				err := ftr.ReadRec(&frec)
				if err == io.EOF {
					break
				} else if err != nil {
					panic(err)
				}

				// Total traffic per minute
				counts[ftr.Inum()]++

				// TCP and UDP traffic per minute
				if frec.Protocol == 6 {
					tcpCounts[ftr.Inum()]++
				} else if frec.Protocol == 17 {
					udpCounts[ftr.Inum()]++
				}

				// Unique sources per minute
				s := int(frec.SrcIP)
				if !sourcesSeen[s] {
					sources[ftr.Inum()]++
					sourcesSeen[s] = true
				}

				// Ports per minute
				ports[ftr.Inum()][frec.DstPort]++
			}

			err = ftr.ReadClassTail()
			if err == io.EOF {
				break
			} else if err != nil {
				panic(err)
			}
		}

		err = ftr.ReadIntervalTail()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
	}

	outname := path.Base(fname)
	outname = strings.Replace(outname, "ucsd-nt.anon.", "", 1)
	outname = strings.Replace(outname, "flowtuple.cors.gz", "csv", 1)
	out, err := os.Create(outname)
	if err != nil {
		panic(err)
	}
	defer out.Close()
	for k := 0; k < 60; k++ {
		out.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n", k, counts[k], sources[k], udpCounts[k], tcpCounts[k]))
	}

	outname = path.Base(fname)
	outname = strings.Replace(outname, "ucsd-nt.anon.", "", 1)
	outname = strings.Replace(outname, "flowtuple.cors.gz", "dports.csv.gz", 1)
	out, err = os.Create(outname)
	if err != nil {
		panic(err)
	}
	defer out.Close()
	w := gzip.NewWriter(out)
	defer w.Close()
	for k := 0; k < 60; k++ {
		var x []string
		for _, y := range ports[k] {
			x = append(x, fmt.Sprintf("%d", y))
		}
		w.Write([]byte(strings.Join(x, ",")))
		w.Write([]byte("\n"))
	}
}
