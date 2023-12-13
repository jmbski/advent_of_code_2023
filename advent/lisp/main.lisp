 (load "C:/HOME/quicklisp/setup.lisp")
(ql:quickload :cl-ppcre) ; Load cl-ppcre library

(defun _print_space () 
    (write-char #\SPACE)
)

(defun _print (str &rest args) "Convenience function to print content without having to manually write out a format function"
    (write str)
    (loop for i below (length args)
        do
        (write (nth i args))
        (_print_space)
    )
)

(defun _printline (val) 
    (write val)
    (terpri)
)

(defun stringify (value)
    (cond
        ((stringp value) (concatenate 'string "\"" value "\""))
        ((numberp value) (format nil "~a" value))
        ((listp value) (format nil "~a" value))
        ((symbolp value) (format nil "~a" value))
        ((null value) "null")
        (t (format nil "~a" value))
    )
)
;(setq my-alist '((key1 . value1) (key2 . value2) (key3 . value3)))
(defvar num-strings 
    '(
        (one . 1)
        (two . 2)
        (three . 3)
        (four . 4)
        ("five" . 5)
        (six . 6)
        (seven . 7)
        (eight . 8)
        (nine . 9)
    )
)

(defun get-num-strings-value (str)
    (_print (type-of str))
    (cdr (assoc str num-strings))
)

(defun is-numeric-string (string)
    (handler-case
        (progn
            (parse-integer string)
            T
        )                       ; Return true if parsing is successful
        (error () 
            nil
        )
    )
)   

(defun as-digit (value)
    (cond 
        ((numberp value) value)
        ((is-numeric-string value) (parse-integer value))
        ((stringp value) 
            (get-num-strings-value value)
        )
    )
)

(defun find-all-digits (string)
    (let* ( (pattern "(?=([\d]|one|two|three|four|five|six|seven|eight|nine))") (matches (cl-ppcre:all-matches-as-strings pattern string)) )
        (mapcar #'as-digit matches)
    )
)

(defun combine-integers (num1 num2)
    (+ (* num1 10) num2)
)

(defun get-line-digits (str-value)
    (let ( (digits (find-all-digits str-value)) )
        (let ( (len (length digits)) )
            (cond 
                ((null digits) 0)
                ((= len 1) 
                    (funcall 
                        ( lambda (digit) 
                            (combine-integers digit digit )  
                        )
                        (first digits)
                    )
                )
                ((>= len 2) 
                    (funcall 
                        ( lambda (_first _last) 
                            (combine-integers _first _last) 
                        )
                        (first digits) (car (last digits) )
                    )
                )
            )
        )
    )
)

(defun read-file-sums ()
    (let ((in (open "../data_files/day1_data.txt" :if-does-not-exist nil)) (sum 0))
        (when in
            (loop for line = (read-line in nil)
                while line do 
                    (setq sum (+ sum (get-line-digits line)))
            )
            (close in)
        )
        (_print "total sum: " sum)
    )
)        ; Return false if an error occurs

(defun some-funct (value)
    (if (is-numeric-string value)
        (some-int-funct value)
        (some-other-funct value)
    )
)

(defun some-int-funct (value)
    ( format *standard-output* "Value is an integer: ~a~%" (parse-integer value) )
)

(defun some-other-funct (value)
    ( format *standard-output* "Value is not an integer: ~a~%" value )
)

;(write (get-line-digits "sg7asdgjsdg9"))
(read-file-sums)
;(write (assoc "five" num-strings ))
;(write (get-num-strings-value (nth 4 num-strings) "five"))