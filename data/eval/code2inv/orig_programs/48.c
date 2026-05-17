void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int c;
int n;
(c = 0);
assume((n > 0));
while (__VERIFIER_nondet_int()) {
{
if (__VERIFIER_nondet_int()) {
if ((c != n)) {
(c = (c + 1));
}
} else {
if ((c == n)) {
(c = 1);
}
}
}
}
if ((c == n)) {
assert((n > -1));
}
}