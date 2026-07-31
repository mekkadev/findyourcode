package billing

import (
	"errors"
	"time"
)

type Invoice struct {
	ID       string
	AmountEU int64
	Due      time.Time
}

// Collect pulls money for an invoice through the card processor and
// retries once when the processor reports a temporary decline.
func Collect(inv Invoice, card Card) (Receipt, error) {
	if inv.AmountEU <= 0 {
		return Receipt{}, errors.New("nothing to collect")
	}
	receipt, err := processor.Capture(card.Token, inv.AmountEU)
	if isTemporary(err) {
		receipt, err = processor.Capture(card.Token, inv.AmountEU)
	}
	if err != nil {
		return Receipt{}, err
	}
	ledger.Append(inv.ID, receipt)
	return receipt, nil
}

func Refund(receipt Receipt, amount int64) error {
	if amount > receipt.Amount {
		return errors.New("refund exceeds original")
	}
	return processor.Reverse(receipt.ID, amount)
}
