void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int lock;
int v1;
int v2;
int v3;
int x;
int y;
(x = y);
(lock = 1);
while ((x != y)) {
{
if (__VERIFIER_nondet_int()) {
{
(lock = 1);
(x = y);
}
} else {
{
(lock = 0);
(x = y);
(y = (y + 1));
}
}
}
}
assert((lock == 1));
}