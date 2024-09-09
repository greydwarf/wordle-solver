use circular_buffer::CircularBuffer;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::collections::HashMap;

const WORD_FREQUENCIES_FILENAME:&str ="data/wordle_words_freqs_full.txt";
const MAX_SCORE:usize = 3_usize.pow(5);

fn main() {
    let freq_entries = read_words();
    let mut remaining_candidates = build_remaining_candidates(&freq_entries);
    filter_candidates(&to_word("tares"), to_ternary("bybyb"), &mut remaining_candidates);
    filter_candidates(&to_word("colin"), to_ternary("ybbbb"), &mut remaining_candidates);
    filter_candidates(&to_word("psych"), to_ternary("bbyyb"), &mut remaining_candidates);
    for guess in remaining_candidates.keys() {
        print!("{}\n", from_word(&guess));
    }
    print!("*** AFTER TARES APPLIED ***\n");
    let best_guesses = compute_best_guesses(&freq_entries, &remaining_candidates);
    for (&guess, entropy) in best_guesses.iter() {
        print!("{} {}\n", entropy, from_word(&guess));
    }
}

// The output is wrapped in a Result to allow matching on errors.
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn read_words() -> Vec<([char; 5], f64)> {
    let mut ret = Vec::new();
    if let Ok(lines) = read_lines(WORD_FREQUENCIES_FILENAME) {
        // Consumes the iterator, returns an (Optional) String
        for line in lines.flatten() {
            let parts = line.split(" ");
            ret.push(compute_overall_freq(parts));
        }
        ret.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    }
    ret
}

fn compute_best_guesses<'a>(freq_entries: &'a Vec<([char; 5], f64)>, remaining_candidates: &HashMap<&[char; 5], f64>) -> Vec<(&'a [char; 5], f64)> {
    let mut ret:Vec<(&[char; 5], f64)> = Vec::new();
    for (guess, _) in freq_entries.iter() {
        let guess_power = compute_guess_power(&guess, &remaining_candidates);
        if guess_power > 0.0 {
            ret.push( (guess, guess_power) );
        }
    }
    ret.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    ret
}

fn build_remaining_candidates(freq_entries: &Vec<([char;5], f64)>) -> HashMap<&[char; 5], f64> {
    let mut remaining_candidates:HashMap<&[char; 5], f64> = HashMap::new();
    for x in freq_entries.iter() {
        let likelihood = sigmoid(quadratic_curve_fit(x.1));
        if likelihood > 0.0 { remaining_candidates.insert(&x.0, likelihood); }
    }
    remaining_candidates
}

fn filter_candidates(word: &[char;5], score: u16, candidates: &mut HashMap<&[char; 5], f64>) {
    candidates.retain(|candidate, _freq| score_guess(word, candidate) == score);
}



fn compute_overall_freq<'a>(mut line: impl Iterator<Item = &'a str>) -> ([char; 5], f64) {
    let word:String = line.next().expect("there was no word here").to_string();
    let mut buf = CircularBuffer::<5, f64>::new();
    for val in line {
        let freq = val.parse().expect("Error parsing the freq");
        buf.push_back(freq);
    }
    let size = buf.len();
    return (to_word(&word), buf.iter().sum::<f64>()/(size as f64));
}

fn quadratic_curve_fit(x:f64) -> f64 {
    return  -19970122538.988*x*x + 41168735.495139*x - 10_f64;
}

fn sigmoid(x: f64) ->f64 {
    return 1_f64 / (1f64 + (-x).exp());
}

fn to_ternary(result: &str) -> u16 {
    let mut ret = 0;
    for ch in result.chars() {
        let mut digit = 0;
        if ch == 'y' {
            digit = 1;
        } else if ch == 'g' {
            digit = 2;
        }
        ret = ret*3 + digit;
    }
    ret
}

fn from_ternary(mut t: u16) -> [char; 5] {
    let mut ret = ['b', 'b', 'b', 'b', 'b'];
    let mut pos = 5;
    while t > 0 {
        pos-=1;
        let digit = t % 3;
        if digit == 1 {
            ret[pos] = 'y';
        } else if digit == 2 {
            ret[pos] = 'g';
        } else {
            ret[pos] = 'b';
        }
        t /= 3;
    }
    ret
}

fn to_word(s: &str) -> [char; 5] {
    let mut ret = [' ', ' ', ' ', ' ', ' '];
    for (i, ch) in s.chars().enumerate() {
        ret[i] = ch;
    }
    ret
}

fn from_word(w: &[char;5]) -> String {
    w.iter().collect()
}

fn score_guess(guess: &[char;5], candidate: &[char;5]) -> u16 {
    let mut ret:u16 = 0;
    let mut g = guess.clone();
    let mut c = candidate.clone();

    for idx in 0..g.len() {
        if g[idx] == c[idx] {
            g[idx] = ':';
            c[idx] = '?';
            ret += 3_u16.pow(4_u32 - idx as u32) * 2;
        }

    }
    for idx in 0..g.len() {
        for cidx in 0..c.len() {
            if c[cidx] == g[idx] {
                c[cidx] = '?';
                ret += 3_u16.pow(4_u32 - idx as u32);
                break;
            }
        }
    }

    ret
}

fn compute_guess_power(guess: &[char; 5], frequencies: &HashMap<&[char; 5], f64>) -> f64 {
    let scores = score_against_dictionary(guess, frequencies);
    let entropy = scores_to_entropy(&scores, frequencies);
    entropy
}

fn scores_to_entropy(scores: &[i32; MAX_SCORE], frequencies: &HashMap<&[char; 5], f64>) -> f64 {
    let mut entropy = 0.0;
    for score in scores {
        let guess_prob = *score as f64 / frequencies.len() as f64;
        let information = if guess_prob == 0.0 {0.0} else {guess_prob * guess_prob.log2()};
        entropy -= information;
    }
    entropy
}
fn score_against_dictionary(guess: &[char; 5], dictionary: &HashMap<&[char; 5], f64>) -> [i32; MAX_SCORE] {
    let mut ret = [0; MAX_SCORE];
    for word in dictionary.keys() {
        let word_score = score_guess(guess, word);
        ret[word_score as usize] += 1;
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score() {
        let a = to_word("aaemp");
        let b = to_word("maaph");
        let c = to_word("mappa");

        assert_eq!(from_ternary(score_guess(&a, &a)), ['g', 'g', 'g', 'g', 'g']);
        assert_eq!(from_ternary(score_guess(&a, &b)), ['y', 'g', 'b', 'y', 'y']);
        assert_eq!(from_ternary(score_guess(&a, &c)), ['y', 'g', 'b', 'y', 'y']);
    }
}
