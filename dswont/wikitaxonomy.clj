;; Copyright 2014 University of Trento, Italy.
;;
;;    Licensed under the Apache License, Version 2.0 (the "License");
;;    you may not use this file except in compliance with the License.
;;    You may obtain a copy of the License at
;;
;;        http://www.apache.org/licenses/LICENSE-2.0
;;
;;    Unless required by applicable law or agreed to in writing, software
;;    distributed under the License is distributed on an "AS IS" BASIS,
;;    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;;    See the License for the specific language governing permissions and
;;    limitations under the License.

(import '[com.hp.hpl.jena.rdf.model Model ModelFactory Resource SimpleSelector])
(import '[com.hp.hpl.jena.ontology OntModelSpec])
(import '[[[com.hp.hpl.jena.vocabulary RDF RDFS]]])
(require '[clojure.java.io :as io])

(defn read-model [file-name]
  (with-open [input (io/input-stream (io/file file-name))]
    (doto (ModelFactory/createOntologyModel OntModelSpec/RDFS_MEM_TRANS_INF)
      (.read input nil))))

(use 'clojure.reflect)

(defn get-classes [m]
    (iterator-seq (.listClasses m)))

(defn get-inds [x]
    (iterator-seq (.listIndividuals x)))

(defn get-label [x]
    (-> x
        (.listPropertyValues RDFS/label)
        iterator-seq
        first .getValue))

(defn get-subcl [c]
    (-> c .listSubClasses iterator-seq))

(defn get-supercl [c]
    (-> c .listSuperClasses iterator-seq))

(defn save-node-types [model file-name]
    (with-open [out (io/writer (io/file file-name))]
        (doseq [cl (get-classes m)]
            (.write out (get-label cl))
            (.write out " class")
            (.newLine out))
        (doseq [cl (get-inds m)]
            (.write out (get-label cl))
            (.write out " individual")
            (.newLine out))))

(defn save-rel-types [model file-name]
    (with-open [out (io/writer (io/file file-name))]
        (doseq [cl (get-classes m)
                subcl (get-subcl cl)]
            (.write out (get-label cl))
            (.write out " -> ")
            (.write out (get-label subcl))
            (.newLine out))
        (doseq [cl (get-classes m)
                ind (get-inds cl)]
            (.write out (get-label cl))
            (.write out " -> ")
            (.write out (get-label ind))
            (.newLine out))))


;; (def m (read-model "/Users/dmirylenka/data/wikitaxonomy/wikipediaOntology.owl"))

;; (save-node-types m "/Users/dmirylenka/data/wikitaxonomy/node-types.txt")

;; (save-rel-types m "/Users/dmirylenka/data/wikitaxonomy/rel-types.txt")
