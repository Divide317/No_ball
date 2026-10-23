class CricketRuleEngine:
    """
    Handles the rule-based side of no-ball detection.
    Instead of using a neural network to count fielders or track bouncer counts,
    we check those directly against the ICC/MCC playing conditions.
    """

    def __init__(self):
        pass

    def get_powerplay_info(self, match_format: str, over: int, ball_index_in_innings: int):
        # Clean up the format string so minor case differences don't break checks
        fmt = match_format.upper().strip()

        if fmt in ["T20", "T20I", "IPL"]:
            # First 6 overs (0 to 5 in 0-indexed) are the mandatory powerplay
            max_bouncers = 2 if fmt == "IPL" else 1
            if over < 6:
                return {
                    "phase": "Powerplay (Overs 1-6)",
                    "max_outside_circle": 2,
                    "max_bouncers": max_bouncers,
                }
            return {
                "phase": "Middle / Death Overs (Overs 7-20)",
                "max_outside_circle": 5,
                "max_bouncers": max_bouncers,
            }

        elif fmt == "ODI":
            # Men's ODI uses three powerplay blocks across the 50 overs
            if over < 10:
                return {"phase": "Powerplay 1 (Overs 1-10)", "max_outside_circle": 2, "max_bouncers": 2}
            elif 10 <= over < 40:
                return {"phase": "Powerplay 2 (Overs 11-40)", "max_outside_circle": 4, "max_bouncers": 2}
            else:
                return {"phase": "Powerplay 3 (Overs 41-50)", "max_outside_circle": 5, "max_bouncers": 2}

        elif fmt == "100-BALL":
            # In The Hundred, the powerplay is the first 25 balls of the innings
            if ball_index_in_innings <= 25:
                return {"phase": "Powerplay (Balls 1-25)", "max_outside_circle": 2, "max_bouncers": 1}
            return {"phase": "Standard Play (Balls 26-100)", "max_outside_circle": 5, "max_bouncers": 1}

        else:
            # Test cricket doesn't have 30-yard circle restrictions
            return {"phase": "Test Match Conditions", "max_outside_circle": 9, "max_bouncers": 2}

    def check_delivery(
        self,
        match_format: str,
        over: int,
        ball_index_in_innings: int,
        bouncers_in_over_before: int,
        is_current_ball_bouncer: bool,
        is_beamer: bool,
        fielders_leg_side: int,
        fielders_behind_square: int,
        fielders_outside_circle: int,
    ):
        """
        Checks if the current ball breaks any fielding or bowling quota laws.
        Returns (is_no_ball, reasons).
        """
        reasons = []
        is_no_ball = False
        fmt = match_format.upper().strip()

        # Law 41.7: Any delivery passing above waist height on the full is a no-ball
        if is_beamer:
            is_no_ball = True
            reasons.append("Waist-high full toss (beamer) on the full is illegal (Law 41.7).")

        # Law 28.4: At most 5 fielders are allowed on the leg (on) side
        if fielders_leg_side > 5:
            is_no_ball = True
            reasons.append(
                f"Leg-side fielding limit exceeded: {fielders_leg_side} fielders on leg side (max 5 allowed, Law 28.4)."
            )

        # Law 28.4: At most 2 fielders can stand behind square leg
        if fielders_behind_square > 2:
            is_no_ball = True
            reasons.append(
                f"Behind-square fielding limit exceeded: {fielders_behind_square} fielders behind square leg (max 2 allowed, Law 28.4)."
            )

        # Law 41.6: Check short-pitched bouncer limits for the current over
        if is_current_ball_bouncer:
            total_bouncers = bouncers_in_over_before + 1

            if fmt in ["T20", "T20I"]:
                # Standard T20 internationals only allow 1 bouncer per over
                if total_bouncers > 1:
                    is_no_ball = True
                    reasons.append(
                        f"Bouncer limit exceeded for T20I: {total_bouncers} bouncers in this over (max 1 allowed, Law 41.6)."
                    )
            elif fmt == "IPL":
                # IPL updated their playing conditions to allow 2 bouncers per over
                if total_bouncers > 2:
                    is_no_ball = True
                    reasons.append(
                        f"Bouncer limit exceeded for IPL: {total_bouncers} bouncers in this over (max 2 allowed)."
                    )
            elif fmt in ["TEST", "ODI"]:
                if total_bouncers > 2:
                    is_no_ball = True
                    reasons.append(
                        f"Bouncer limit exceeded for {fmt}: {total_bouncers} bouncers in this over (max 2 allowed, Law 41.6)."
                    )
            elif fmt == "100-BALL":
                if total_bouncers > 1:
                    is_no_ball = True
                    reasons.append(
                        f"Bouncer limit exceeded for 100-ball: {total_bouncers} bouncers in this set (max 1 allowed)."
                    )

        # Check 30-yard circle restrictions based on format and current over
        if fmt in ["T20", "T20I", "IPL"]:
            if over < 6 and fielders_outside_circle > 2:
                is_no_ball = True
                reasons.append(
                    f"Powerplay circle violation: {fielders_outside_circle} fielders outside 30-yard circle (max 2 in Overs 1-6)."
                )
            elif over >= 6 and fielders_outside_circle > 5:
                is_no_ball = True
                reasons.append(
                    f"Fielding circle violation: {fielders_outside_circle} fielders outside 30-yard circle (max 5 in Overs 7-20)."
                )

        elif fmt == "ODI":
            if over < 10 and fielders_outside_circle > 2:
                is_no_ball = True
                reasons.append(
                    f"ODI Powerplay 1 violation: {fielders_outside_circle} fielders outside circle (max 2 allowed)."
                )
            elif 10 <= over < 40 and fielders_outside_circle > 4:
                is_no_ball = True
                reasons.append(
                    f"ODI Powerplay 2 violation: {fielders_outside_circle} fielders outside circle (max 4 allowed)."
                )
            elif over >= 40 and fielders_outside_circle > 5:
                is_no_ball = True
                reasons.append(
                    f"ODI Powerplay 3 violation: {fielders_outside_circle} fielders outside circle (max 5 allowed)."
                )

        elif fmt == "100-BALL":
            if ball_index_in_innings <= 25 and fielders_outside_circle > 2:
                is_no_ball = True
                reasons.append(
                    f"100-ball Powerplay violation: {fielders_outside_circle} fielders outside circle (max 2 allowed)."
                )
            elif ball_index_in_innings > 25 and fielders_outside_circle > 5:
                is_no_ball = True
                reasons.append(
                    f"100-ball fielding violation: {fielders_outside_circle} fielders outside circle (max 5 allowed)."
                )

        return is_no_ball, reasons

    def check_delivery_detailed(
        self,
        match_format: str,
        over: int,
        ball_index_in_innings: int,
        bouncers_in_over_before: int,
        is_current_ball_bouncer: bool,
        is_beamer: bool,
        fielders_leg_side: int,
        fielders_behind_square: int,
        fielders_outside_circle: int,
        overstep_prob: float = 0.0,
        beamer_prob: float = 0.0,
    ):
        """
        Formats all 6 checks into a clean list so the frontend dashboard can show
        which laws passed and which ones failed.
        """
        fmt = match_format.upper().strip()
        pp_info = self.get_powerplay_info(fmt, over, ball_index_in_innings)
        total_bouncers = bouncers_in_over_before + (1 if is_current_ball_bouncer else 0)
        max_bouncers = pp_info["max_bouncers"]
        max_outside = pp_info["max_outside_circle"]

        checks = [
            {
                "id": "front_foot",
                "law": "MCC Law 21.5",
                "title": "Front-Foot Popping Crease",
                "passed": overstep_prob <= 0.50,
                "metric": f"{overstep_prob * 100:.1f}% overstep risk",
                "limit": "Some part of foot behind line",
                "source": "Crease CNN",
            },
            {
                "id": "beamer",
                "law": "MCC Law 41.7",
                "title": "Waist-High Full Toss (Beamer)",
                "passed": not is_beamer,
                "metric": f"{beamer_prob * 100:.1f}% beamer risk"
                if beamer_prob > 0
                else ("Flagged" if is_beamer else "Below waist"),
                "limit": "Below batter's waist on the full",
                "source": "Waist CNN + Manual",
            },
            {
                "id": "bouncer_quota",
                "law": "MCC Law 41.6",
                "title": "Short-Pitched Bouncer Quota",
                "passed": not (is_current_ball_bouncer and total_bouncers > max_bouncers),
                "metric": f"{total_bouncers} / {max_bouncers} in over",
                "limit": f"Max {max_bouncers} per over ({fmt})",
                "source": "Rule Engine",
            },
            {
                "id": "powerplay_circle",
                "law": "ICC Clause 28.7",
                "title": f"30-Yard Circle ({pp_info['phase']})",
                "passed": True if fmt == "TEST" else (fielders_outside_circle <= max_outside),
                "metric": "No restriction (Test)"
                if fmt == "TEST"
                else f"{fielders_outside_circle} / {max_outside} outside",
                "limit": "Unrestricted" if fmt == "TEST" else f"Max {max_outside} outside circle",
                "source": "Rule Engine",
            },
            {
                "id": "leg_side",
                "law": "MCC Law 28.4",
                "title": "On-Side (Leg-Side) Fielders",
                "passed": fielders_leg_side <= 5,
                "metric": f"{fielders_leg_side} / 5 on leg side",
                "limit": "Max 5 fielders on leg side",
                "source": "Rule Engine",
            },
            {
                "id": "behind_square",
                "law": "MCC Law 28.4",
                "title": "Behind Popping Crease (Square Leg)",
                "passed": fielders_behind_square <= 2,
                "metric": f"{fielders_behind_square} / 2 behind square",
                "limit": "Max 2 behind square leg",
                "source": "Rule Engine",
            },
        ]

        # In white-ball formats, any no-ball results in a free hit on the next ball
        any_failed = any(not c["passed"] for c in checks)
        free_hit = any_failed and fmt in ["T20", "T20I", "IPL", "ODI", "100-BALL"]

        return {
            "powerplay_phase": pp_info["phase"],
            "max_outside_circle": max_outside,
            "max_bouncers_per_over": max_bouncers,
            "free_hit_awarded": free_hit,
            "checks": checks,
        }


if __name__ == "__main__":
    re = CricketRuleEngine()
    nb, reasons = re.check_delivery(
        match_format="T20",
        over=3,
        ball_index_in_innings=18,
        bouncers_in_over_before=0,
        is_current_ball_bouncer=False,
        is_beamer=False,
        fielders_leg_side=4,
        fielders_behind_square=1,
        fielders_outside_circle=3,
    )
    print("Powerplay check:", "NO-BALL" if nb else "LEGAL", reasons)
